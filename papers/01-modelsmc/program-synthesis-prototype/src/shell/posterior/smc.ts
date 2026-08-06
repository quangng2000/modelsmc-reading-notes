import { jsonStringify, programToJsonValue } from "../ast/render.js";
import type { RandomSource } from "../engine/random.js";
import { effectiveSampleSize, normalizeLogWeights } from "../engine/numerics.js";
import { systematicResample } from "../engine/resampling.js";
import { createPriorSampler, type PriorSampler } from "../prior/index.js";
import { scoreCalibrated } from "../scoring/calibrated.js";
import {
  drawAccepted,
  evaluateSidecarDraw,
  feedbackPrompt,
  proposalPrompt,
  type AcceptedProposal,
  type ProposalTask,
  type RawDraw,
} from "./llm-proposal.js";
import type { SidecarClient } from "./sidecar-client.js";

/**
 * Calibrated tempered Sequential Monte Carlo over programs.
 *
 * Target sequence (likelihood tempering): pi_gamma(m) proportional to
 * exp(-beta*cost(m)) * L(m)^gamma with gamma: 0 -> 1 — prior at gamma 0,
 * posterior at gamma 1. Per stage: AIS reweighting by L^(dgamma), systematic
 * resampling on ESS collapse, then Metropolis-Hastings rejuvenation sweeps
 * with an independence proposal. MH moves need no weight update (the kernel
 * leaves pi_gamma invariant), but they DO need the proposal density at the
 * current point — which dictates the two exactly-computable variants:
 *
 * - moves = "prior": proposals from the exact prior sampler. The density is
 *   known in closed form for EVERY program, the acceptance ratio reduces to
 *   the tempered likelihood ratio, and no LLM is involved. The coverage
 *   kernel.
 * - moves = "llm": proposals from the LLM (unconstrained draws, canonical
 *   rejection, retry-until-accept realizing q/P(accept)). Initialization is
 *   also from the LLM with importance weights p~(m)/q(m) targeting the
 *   prior, so every particle's proposal density is the one cached at its own
 *   generation — always valid, because prior moves never run on this island.
 *   The concentration kernel (v1 EOS caveat applies).
 *
 * Mixing the two kernels on one particle trajectory is NOT implemented: an
 * accepted prior move invalidates the cached LLM density, and re-weighting
 * around that (a Del Moral refresh with backward kernel pi_gamma) introduces
 * a normalizer Z_gamma only on the refreshed branch, which does not cancel
 * across particles. Exact alternation therefore needs the proposal density
 * of arbitrary programs — a teacher-forcing scoring call the current serving
 * lacks. Run both islands and combine estimates instead (two consistent
 * estimators of the same posterior).
 */

export interface SmcParticle {
  proposal: AcceptedProposal;
  logWeight: number;
}

export interface CalibratedSmcOptions {
  readonly task: ProposalTask;
  readonly particleCount: number;
  /** Tempering schedule; strictly increasing, ending at exactly 1. */
  readonly gammas: readonly number[];
  readonly moves: "prior" | "llm" | "llm-feedback";
  readonly sweepsPerStage: number;
  readonly essThresholdRatio: number;
  readonly rng: RandomSource;
  readonly drawer?: () => Promise<RawDraw>;
  /** Required for llm-feedback: exact generate/score/encode primitives. */
  readonly sidecar?: SidecarClient;
  /** Generation length cap for sidecar draws (a capped draw is just a rejected move). */
  readonly generateMaxTokens?: number;
  /** Proposal temperature (tempered softmax, exact density); score calls use the same T. */
  readonly proposalTemperature?: number;
  readonly onReject?: (reason: string) => void;
  readonly trace?: (message: string) => void;
}

export interface StageDiagnostics {
  readonly gamma: number;
  readonly relativeEss: number;
  readonly resampled: boolean;
  readonly acceptRate: number;
  readonly uniquePrograms: number;
}

export interface CalibratedSmcResult {
  readonly particles: readonly SmcParticle[];
  readonly normalizedWeights: readonly number[];
  readonly diagnostics: readonly StageDiagnostics[];
  readonly llmDraws: number;
}

function temperedLogTarget(proposal: AcceptedProposal, gamma: number): number {
  return proposal.logPriorUnnormalized + gamma * proposal.logLikelihood;
}

function priorAsProposal(sampler: PriorSampler, task: ProposalTask): AcceptedProposal {
  const { program } = sampler.sample();
  const score = scoreCalibrated(program, {
    inputType: task.inputType,
    outputType: task.outputType,
    examples: task.examples,
    beta: task.beta,
    noise: task.noise,
  });
  if ("rejected" in score) throw new Error(`prior sample rejected: ${score.rejected}`);
  return {
    program,
    rendered: "",
    cost: score.cost,
    logLikelihood: score.logLikelihood,
    logPriorUnnormalized: score.logPriorUnnormalized,
    logProposal: Number.NaN, // never used on the prior island
  };
}

export async function runCalibratedSmc(options: CalibratedSmcOptions): Promise<CalibratedSmcResult> {
  const { task, particleCount, gammas, rng } = options;
  if (gammas.length === 0 || Math.abs(gammas[gammas.length - 1]! - 1) > 1e-12) {
    throw new Error("gamma schedule must end at 1");
  }
  for (let index = 1; index < gammas.length; index += 1) {
    if (!(gammas[index]! > gammas[index - 1]!)) throw new Error("gammas must be strictly increasing");
  }
  const trace = options.trace ?? (() => {});
  const priorSampler = createPriorSampler({
    inputType: task.inputType,
    outputType: task.outputType,
    costCap: task.costCap,
    constants: task.integerConstants,
    beta: task.beta,
    rng,
  });
  let llmDraws = 0;
  const countingDrawer =
    options.drawer === undefined
      ? undefined
      : async () => {
          llmDraws += 1;
          return options.drawer!();
        };
  if (options.moves === "llm" && countingDrawer === undefined) {
    throw new Error("llm moves require a drawer");
  }
  const sidecar = options.sidecar;
  if (options.moves === "llm-feedback" && sidecar === undefined) {
    throw new Error("llm-feedback moves require a sidecar client");
  }
  const basePrompt = proposalPrompt(task);

  const generateMaxTokens = options.generateMaxTokens ?? 700;
  const proposalTemperature = options.proposalTemperature ?? 1;
  async function sidecarAccepted(prompt: string): Promise<AcceptedProposal> {
    for (let attempt = 0; attempt < 200; attempt += 1) {
      llmDraws += 1;
      const outcome = await evaluateSidecarDraw(await sidecar!.generate(prompt, generateMaxTokens, proposalTemperature), task, sidecar!.encode);
      if ("rejected" in outcome) {
        options.onReject?.(outcome.rejected);
        continue;
      }
      return outcome;
    }
    throw new Error("no accepted sidecar proposal in 200 attempts");
  }

  // Initialization targeting pi_0 = prior.
  const particles: SmcParticle[] = [];
  for (let index = 0; index < particleCount; index += 1) {
    if (options.moves === "llm") {
      const proposal = await drawAccepted(countingDrawer!, task, options.onReject);
      particles.push({
        proposal,
        logWeight: proposal.logPriorUnnormalized - proposal.logProposal,
      });
    } else if (options.moves === "llm-feedback") {
      // Fixed base prompt: the acceptance-conditioning constant is shared by
      // every init draw and cancels in self-normalization.
      const proposal = await sidecarAccepted(basePrompt);
      particles.push({
        proposal,
        logWeight: proposal.logPriorUnnormalized - proposal.logProposal,
      });
    } else {
      particles.push({ proposal: priorAsProposal(priorSampler, task), logWeight: 0 });
    }
  }

  const diagnostics: StageDiagnostics[] = [];
  let previousGamma = 0;

  for (const gamma of gammas) {
    // AIS reweight from pi_{previous} to pi_{gamma}.
    const deltaGamma = gamma - previousGamma;
    for (const particle of particles) {
      particle.logWeight += deltaGamma * particle.proposal.logLikelihood;
    }
    previousGamma = gamma;

    // Resample on ESS collapse.
    const normalized = normalizeLogWeights(particles.map((particle) => particle.logWeight));
    const relativeEss = effectiveSampleSize(normalized.weights) / particleCount;
    let resampled = false;
    if (relativeEss < options.essThresholdRatio) {
      const ancestors = systematicResample(normalized.weights, rng);
      const snapshot = ancestors.map((ancestor) => particles[ancestor]!);
      snapshot.forEach((source, index) => {
        particles[index] = { proposal: source.proposal, logWeight: 0 };
      });
      resampled = true;
    }

    // Independence-MH rejuvenation sweeps (no weight change: pi_gamma-invariant).
    let proposalsMade = 0;
    let accepts = 0;
    for (let sweep = 0; sweep < options.sweepsPerStage; sweep += 1) {
      for (const particle of particles) {
        let candidate: AcceptedProposal;
        let logAlpha: number;
        if (options.moves === "prior") {
          candidate = priorAsProposal(priorSampler, task);
          // [pi(c)/pi(x)] * [p(x)/p(c)] = (L(c)/L(x))^gamma.
          logAlpha = gamma * (candidate.logLikelihood - particle.proposal.logLikelihood);
        } else if (options.moves === "llm-feedback") {
          // Feedback-conditioned independence MH. ONE raw draw per move — no
          // retry: the acceptance-conditioning constant would depend on the
          // per-particle prompt and would not cancel in the ratio. An invalid
          // draw proposes a zero-target point, i.e. the move is rejected.
          proposalsMade += 1;
          llmDraws += 1;
          const promptForCurrent = feedbackPrompt(task, particle.proposal);
          const outcome = await evaluateSidecarDraw(
            await sidecar!.generate(promptForCurrent, generateMaxTokens, proposalTemperature),
            task,
            sidecar!.encode,
          );
          if ("rejected" in outcome) {
            options.onReject?.(outcome.rejected);
            continue;
          }
          llmDraws += 1; // the reverse-scoring pass is a full model evaluation too
          const reverseLogQ = await sidecar!.score(
            feedbackPrompt(task, outcome),
            particle.proposal.tokenIds!,
            proposalTemperature,
          );
          logAlpha =
            temperedLogTarget(outcome, gamma) +
            reverseLogQ -
            (temperedLogTarget(particle.proposal, gamma) + outcome.logProposal);
          if (Math.log(rng.next()) < logAlpha) {
            particle.proposal = outcome;
            accepts += 1;
          }
          continue;
        } else {
          candidate = await drawAccepted(countingDrawer!, task, options.onReject);
          logAlpha =
            temperedLogTarget(candidate, gamma) -
            candidate.logProposal -
            (temperedLogTarget(particle.proposal, gamma) - particle.proposal.logProposal);
        }
        proposalsMade += 1;
        if (Math.log(rng.next()) < logAlpha) {
          particle.proposal = candidate;
          accepts += 1;
        }
      }
    }

    const unique = new Set(
      particles.map((particle) => jsonStringify(programToJsonValue(particle.proposal.program))),
    ).size;
    const stage: StageDiagnostics = {
      gamma,
      relativeEss,
      resampled,
      acceptRate: proposalsMade === 0 ? 0 : accepts / proposalsMade,
      uniquePrograms: unique,
    };
    diagnostics.push(stage);
    trace(
      `gamma=${gamma.toFixed(3)} relESS=${relativeEss.toFixed(3)}${resampled ? " resampled" : ""} ` +
        `acc=${(stage.acceptRate * 100).toFixed(0)}% unique=${unique}/${particleCount}`,
    );
  }

  const finalNormalized = normalizeLogWeights(particles.map((particle) => particle.logWeight));
  return { particles, normalizedWeights: finalNormalized.weights, diagnostics, llmDraws };
}

/** Linear tempering schedule of the given length ending at 1. */
export function linearGammas(stages: number): number[] {
  if (!Number.isSafeInteger(stages) || stages < 1) throw new Error("stages must be a positive integer");
  return Array.from({ length: stages }, (_unused, index) => (index + 1) / stages);
}
