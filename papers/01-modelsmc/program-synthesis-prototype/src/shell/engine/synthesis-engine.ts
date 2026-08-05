import {
  acceptProgram,
  type Program,
} from "../../core/language.verify.js";
import type { ExperimentConfig } from "../config/index.js";
import { deriveSynthesisTrace } from "../deduction/index.js";
import type { Proposer } from "../proposal/index.js";
import { renderExpr, renderType, renderValue, exprToJsonValue } from "../ast/render.js";
import {
  scoreProgram,
  type ValidScore,
} from "../scoring/index.js";
import { effectiveSampleSize, formatNumber, normalizeLogWeights } from "./numerics.js";
import {
  bestChampion,
  bestParticle,
  betterChampion,
  initialExpressions,
  particleLineage,
  scoringOptions,
} from "./population.js";
import { traceParticleScore } from "./trace-particle.js";
import type { IterationSummary, Particle, SynthesisResult } from "./types.js";
import { propagatePopulation } from "./propagation.js";
import { SeededRandom } from "./random.js";
import { systematicResample } from "./resampling.js";
import { NullTraceSink, type TraceSink } from "./trace.js";

export interface EngineOptions {
  readonly config: ExperimentConfig;
  readonly proposer: Proposer;
  readonly trace?: TraceSink;
}

export class SynthesisEngine {
  private readonly config: ExperimentConfig;
  private readonly proposer: Proposer;
  private readonly trace: TraceSink;
  private readonly random: SeededRandom;
  private nextParticleId = 0;
  private proposalCalls = 0;

  constructor(options: EngineOptions) {
    this.config = options.config;
    this.proposer = options.proposer;
    this.trace = options.trace ?? new NullTraceSink();
    this.random = new SeededRandom(options.config.seed);
  }

  private allocateId(): number {
    const id = this.nextParticleId;
    this.nextParticleId += 1;
    return id;
  }

  private scoreOrThrow(expression: Program): ValidScore {
    const score = scoreProgram(expression, scoringOptions(this.config));
    if (score.kind === "Rejected") {
      throw new Error(`internal seed program was rejected: ${score.reason}`);
    }
    return score;
  }

  private initialize(): Particle[] {
    const seeds = initialExpressions(this.config);
    if (seeds.length === 0) throw new Error("no valid initial expressions are available");
    const raw = Array.from({ length: this.config.particles }, (_unused, index) => {
      const expression = seeds[index % seeds.length]!;
      return {
        id: this.allocateId(),
        parentId: null,
        expression,
        score: this.scoreOrThrow(expression),
        weight: 0,
        origin: "initial" as const,
        rationale: "typed terminal seed",
      };
    });
    return this.normalizePopulation(raw, "initial");
  }

  private normalizePopulation(
    particles: readonly Omit<Particle, "weight">[] | readonly Particle[],
    phase: string,
  ): Particle[] {
    const normalized = normalizeLogWeights(particles.map((particle) => particle.score.logTarget));
    const weighted = particles.map((particle, index): Particle => ({
      ...particle,
      weight: normalized.weights[index]!,
    }));
    this.trace.emit(
      "weights.normalized",
      `${phase} normalized weights: [${weighted.map((particle) => formatNumber(particle.weight)).join(", ")}]${normalized.usedUniformFallback ? " (uniform fallback)" : ""}`,
      {
        phase,
        weights: weighted.map((particle) => particle.weight),
        uniformFallback: normalized.usedUniformFallback,
      },
    );
    for (const particle of weighted) traceParticleScore(this.trace, particle, phase);
    return weighted;
  }

  async run(): Promise<SynthesisResult> {
    this.trace.emit(
      "run.started",
      `loaded ${this.config.examples.length} examples for ${renderType(this.config.inputType)} -> ${renderType(this.config.outputType)}; particles=${this.config.particles} iterations=${this.config.iterations} proposer=${this.proposer.name}`,
      {
        task: this.config.name,
        examples: this.config.examples.length,
        inputType: this.config.inputType,
        outputType: this.config.outputType,
        particles: this.config.particles,
        iterations: this.config.iterations,
        cloneProbability: this.config.cloneProbability,
        relativeEssThreshold: this.config.essThreshold,
        seed: this.config.seed,
        proposer: this.proposer.name,
      },
    );
    this.config.examples.forEach((example, index) => {
      this.trace.emit(
        "spec.example",
        `example ${index + 1}: ${renderValue(example.input)} -> ${renderValue(example.output)}`,
        {
          example: index + 1,
          input: renderValue(example.input),
          output: renderValue(example.output),
        },
      );
    });
    for (const event of deriveSynthesisTrace(
      this.config.inputType,
      this.config.outputType,
      this.config.examples,
    )) {
      this.trace.emit(event.kind, event.message, event.data);
    }

    let particles = this.initialize();
    const archive = new Map<number, Particle>();
    for (const particle of particles) archive.set(particle.id, particle);
    let champion = bestChampion(particles);
    let firstExactIteration: number | null = champion.score.exactProgram ? 0 : null;
    let firstExactProposalCall: number | null = null;
    this.trace.emit(
      "search.champion",
      `best-so-far initialized: particle=${champion.id} ${renderExpr(champion.expression)} loss=${formatNumber(champion.score.totalLoss)} cost=${champion.score.cost}`,
      {
        particleId: champion.id,
        expression: exprToJsonValue(champion.expression),
        logTarget: champion.score.logTarget,
        totalLoss: champion.score.totalLoss,
        cost: champion.score.cost,
        exact: champion.score.exactProgram,
      },
    );
    const summaries: IterationSummary[] = [];

    for (let iteration = 1; iteration <= this.config.iterations; iteration += 1) {
      const essBefore = effectiveSampleSize(particles.map((particle) => particle.weight));
      const relativeEssBefore = essBefore / particles.length;
      const resampled = relativeEssBefore < this.config.essThreshold;
      const ancestors = resampled
        ? systematicResample(
            particles.map((particle) => particle.weight),
            this.random,
          )
        : particles.map((_particle, index) => index);
      this.trace.emit(
        resampled ? "resampling.completed" : "resampling.skipped",
        `iteration ${iteration}: ESS=${formatNumber(essBefore)}/${particles.length} relative=${formatNumber(relativeEssBefore)} threshold=${formatNumber(this.config.essThreshold)}; ${resampled ? `resampled ancestors=[${ancestors.join(", ")}]` : "resampling skipped"}`,
        { iteration, ess: essBefore, relativeEss: relativeEssBefore, resampled, ancestors },
      );

      const propagation = await propagatePopulation({
        config: this.config,
        proposer: this.proposer,
        trace: this.trace,
        random: this.random,
        particles,
        ancestors,
        iteration,
        startingProposalCalls: this.proposalCalls,
        exactAlreadyFound: firstExactIteration !== null,
        allocateId: () => this.allocateId(),
      });
      this.proposalCalls = propagation.proposalCalls;
      if (propagation.firstExact !== null) {
        firstExactIteration = propagation.firstExact.iteration;
        firstExactProposalCall = propagation.firstExact.proposalCall;
      }

      // Score belongs to each current AST. Old particle weights are deliberately not
      // multiplied in again, which would count the same examples repeatedly.
      particles = this.normalizePopulation(propagation.offspring, `iteration-${iteration}`);
      for (const particle of particles) archive.set(particle.id, particle);
      const best = bestParticle(particles);
      const iterationChampion = bestChampion(particles);
      if (betterChampion(iterationChampion, champion)) {
        champion = iterationChampion;
        this.trace.emit(
          "search.champion",
          `best-so-far improved at iteration ${iteration}: particle=${champion.id} ${renderExpr(champion.expression)} loss=${formatNumber(champion.score.totalLoss)} cost=${champion.score.cost}${champion.score.exactProgram ? " exact" : ""}`,
          {
            iteration,
            particleId: champion.id,
            expression: exprToJsonValue(champion.expression),
            logTarget: champion.score.logTarget,
            totalLoss: champion.score.totalLoss,
            cost: champion.score.cost,
            exact: champion.score.exactProgram,
          },
        );
      }
      summaries.push({
        iteration,
        essBefore,
        relativeEssBefore,
        resampled,
        ancestors,
        bestParticleId: best.id,
        bestExpression: best.expression,
        bestWeight: best.weight,
      });
      this.trace.emit(
        "iteration.completed",
        `iteration ${iteration} complete: unique programs=${new Set(particles.map((particle) => renderExpr(particle.expression))).size}/${particles.length}; exact programs=${particles.filter((particle) => particle.score.exactProgram).length}/${particles.length}; best particle=${best.id} ${renderExpr(best.expression)} loss=${formatNumber(best.score.totalLoss)} cost=${best.score.cost} weight=${formatNumber(best.weight)}`,
        {
          iteration,
          uniquePrograms: new Set(particles.map((particle) => renderExpr(particle.expression))).size,
          exactPrograms: particles.filter((particle) => particle.score.exactProgram).length,
          particles: particles.length,
          bestParticleId: best.id,
          bestExpression: exprToJsonValue(best.expression),
          bestLoss: best.score.totalLoss,
          bestCost: best.score.cost,
          bestWeight: best.weight,
          exact: best.score.exactProgram,
        },
      );
    }

    const best = champion;
    const exact = acceptProgram(best.expression, [...this.config.examples]);
    const championLineage = particleLineage(best, archive);
    this.trace.emit(
      "run.completed",
      `finished: best-so-far=${renderExpr(best.expression)} loss=${formatNumber(best.score.totalLoss)} cost=${best.score.cost}; exact=${exact}; lineage losses=[${championLineage.map((particle) => formatNumber(particle.score.totalLoss)).join(" -> ")}]`,
      {
        bestParticleId: best.id,
        bestExpression: exprToJsonValue(best.expression),
        bestLoss: best.score.totalLoss,
        bestCost: best.score.cost,
        bestWeight: best.weight,
        exact,
        proposalCalls: this.proposalCalls,
        firstExactIteration,
        firstExactProposalCall,
        championLineage: championLineage.map((particle) => ({
          particleId: particle.id,
          parentId: particle.parentId,
          expression: exprToJsonValue(particle.expression),
          loss: particle.score.totalLoss,
          exactMatches: particle.score.exactMatches,
          cost: particle.score.cost,
          origin: particle.origin,
        })),
      },
    );
    return {
      particles,
      iterations: summaries,
      best,
      exact,
      proposalCalls: this.proposalCalls,
      firstExactIteration,
      firstExactProposalCall,
      championLineage,
    };
  }
}
