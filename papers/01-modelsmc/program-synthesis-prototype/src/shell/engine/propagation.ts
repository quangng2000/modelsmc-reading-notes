import { inferType } from "../../core/language.verify.js";
import type { ExperimentConfig } from "../config/index.js";
import { exprToJsonValue, renderExpr, renderType } from "../ast/render.js";
import type { Proposer } from "../proposal/index.js";
import { scoreProgram } from "../scoring/index.js";
import { formatNumber } from "./numerics.js";
import { scoringOptions } from "./population.js";
import type { SeededRandom } from "./random.js";
import type { TraceSink } from "./trace.js";
import type { Particle } from "./types.js";

export interface FirstExactDiscovery {
  readonly iteration: number;
  readonly proposalCall: number;
}

export interface PropagationOptions {
  readonly config: ExperimentConfig;
  readonly proposer: Proposer;
  readonly trace: TraceSink;
  readonly random: SeededRandom;
  readonly particles: readonly Particle[];
  readonly ancestors: readonly number[];
  readonly iteration: number;
  readonly startingProposalCalls: number;
  readonly exactAlreadyFound: boolean;
  readonly allocateId: () => number;
}

export interface PropagationResult {
  readonly offspring: readonly Omit<Particle, "weight">[];
  readonly proposalCalls: number;
  readonly firstExact: FirstExactDiscovery | null;
}

/** Move, validate, and score one complete SMC population. */
export async function propagatePopulation(
  options: PropagationOptions,
): Promise<PropagationResult> {
  const offspring: Omit<Particle, "weight">[] = [];
  let proposalCalls = options.startingProposalCalls;
  let firstExact: FirstExactDiscovery | null = null;

  for (let slot = 0; slot < options.particles.length; slot += 1) {
    const ancestorIndex = options.ancestors[slot]!;
    const ancestor = options.particles[ancestorIndex]!;
    const draw = options.random.next();
    if (draw < options.config.cloneProbability) {
      offspring.push({
        id: options.allocateId(),
        parentId: ancestor.id,
        expression: ancestor.expression,
        score: ancestor.score,
        origin: "clone",
        rationale: "clone draw below alpha",
      });
      options.trace.emit(
        "proposal.cloned",
        `iteration ${options.iteration} slot=${slot}: draw=${formatNumber(draw)} < alpha=${formatNumber(options.config.cloneProbability)}; cloned particle ${ancestor.id}`,
        {
          iteration: options.iteration,
          slot,
          draw,
          alpha: options.config.cloneProbability,
          ancestorId: ancestor.id,
        },
      );
      continue;
    }

    const requestIndex = proposalCalls;
    proposalCalls += 1;
    options.trace.emit(
      "proposal.requested",
      `iteration ${options.iteration} slot=${slot}: draw=${formatNumber(draw)} >= alpha=${formatNumber(options.config.cloneProbability)}; asking ${options.proposer.name} to revise particle ${ancestor.id}`,
      {
        iteration: options.iteration,
        slot,
        draw,
        alpha: options.config.cloneProbability,
        ancestorId: ancestor.id,
        requestIndex,
      },
    );

    try {
      const proposal = await options.proposer.propose({
        requestIndex,
        iteration: options.iteration,
        slot,
        avoidPrograms: offspring.map((particle) => particle.expression),
        inputType: options.config.inputType,
        outputType: options.config.outputType,
        examples: options.config.examples,
        integerConstants: options.config.integerConstants,
        maxDepth: options.config.maxDepth,
        maxNodes: options.config.maxNodes,
        maxCost: options.config.maxCost,
        ancestor: ancestor.expression,
        ancestorScore: ancestor.score,
        ancestorFeedback: ancestor.rationale,
      });
      const inferred = inferType(proposal.expression, options.config.inputType);
      options.trace.emit(
        "type.checked",
        `iteration ${options.iteration} slot=${slot}: type check for ${renderExpr(proposal.expression)} -> ${inferred.kind === "TypeOk" ? renderType(inferred.inferred) : "TypeError"}; expected ${renderType(options.config.outputType)}`,
        {
          iteration: options.iteration,
          slot,
          expression: exprToJsonValue(proposal.expression),
          result: inferred.kind,
          inferredType: inferred.kind === "TypeOk" ? inferred.inferred : null,
          expectedType: options.config.outputType,
        },
      );
      const score = scoreProgram(proposal.expression, scoringOptions(options.config));
      if (score.kind === "Rejected") {
        options.trace.emit(
          "proposal.rejected",
          `iteration ${options.iteration} slot=${slot}: proposed ${renderExpr(proposal.expression)}; ${score.reason}; retaining ancestor ${ancestor.id}`,
          {
            iteration: options.iteration,
            slot,
            ancestorId: ancestor.id,
            expression: exprToJsonValue(proposal.expression),
            renderedExpression: renderExpr(proposal.expression),
            inferredType: inferred.kind === "TypeOk" ? inferred.inferred : "TypeError",
            reason: score.reason,
            source: proposal.source,
          },
        );
        offspring.push({
          id: options.allocateId(),
          parentId: ancestor.id,
          expression: ancestor.expression,
          score: ancestor.score,
          origin: "fallback",
          rationale: `proposal rejected: ${score.reason}`,
        });
        continue;
      }

      const childId = options.allocateId();
      options.trace.emit(
        "proposal.accepted",
        `iteration ${options.iteration} slot=${slot}: ${proposal.source} proposed ${renderExpr(proposal.expression)}; inferred ${renderType(score.inferredType)}; accepted for scoring`,
        {
          iteration: options.iteration,
          slot,
          ancestorId: ancestor.id,
          expression: exprToJsonValue(proposal.expression),
          renderedExpression: renderExpr(proposal.expression),
          inferredType: score.inferredType,
          rationale: proposal.rationale,
          source: proposal.source,
        },
      );
      options.trace.emit(
        "proposal.feedback",
        `iteration ${options.iteration} slot=${slot}: parent ${ancestor.id} -> child ${childId}; loss ${formatNumber(ancestor.score.totalLoss)} -> ${formatNumber(score.totalLoss)} (delta ${formatNumber(score.totalLoss - ancestor.score.totalLoss)}); exact examples ${ancestor.score.exactMatches}/${options.config.examples.length} -> ${score.exactMatches}/${options.config.examples.length}; cost ${ancestor.score.cost} -> ${score.cost}; ${proposal.rationale}`,
        {
          iteration: options.iteration,
          slot,
          requestIndex,
          ancestorId: ancestor.id,
          childId,
          previousLoss: ancestor.score.totalLoss,
          nextLoss: score.totalLoss,
          lossDelta: score.totalLoss - ancestor.score.totalLoss,
          previousExactMatches: ancestor.score.exactMatches,
          nextExactMatches: score.exactMatches,
          examples: options.config.examples.length,
          previousCost: ancestor.score.cost,
          nextCost: score.cost,
          exact: score.exactProgram,
          rationale: proposal.rationale,
        },
      );
      if (score.exactProgram && !options.exactAlreadyFound && firstExact === null) {
        firstExact = {
          iteration: options.iteration,
          proposalCall: requestIndex + 1,
        };
        options.trace.emit(
          "search.first-exact",
          `first exact program found at iteration ${options.iteration}, proposal call ${requestIndex + 1}: particle=${childId} ${renderExpr(proposal.expression)}`,
          {
            iteration: options.iteration,
            slot,
            requestIndex,
            proposalCall: requestIndex + 1,
            particleId: childId,
            expression: exprToJsonValue(proposal.expression),
            cost: score.cost,
          },
        );
      }
      offspring.push({
        id: childId,
        parentId: ancestor.id,
        expression: proposal.expression,
        score,
        origin: proposal.source,
        rationale: proposal.rationale,
      });
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      options.trace.emit(
        "proposal.failed",
        `iteration ${options.iteration} slot=${slot}: ${options.proposer.name} failed (${reason}); retaining ancestor ${ancestor.id}`,
        {
          iteration: options.iteration,
          slot,
          ancestorId: ancestor.id,
          reason,
          proposer: options.proposer.name,
        },
      );
      offspring.push({
        id: options.allocateId(),
        parentId: ancestor.id,
        expression: ancestor.expression,
        score: ancestor.score,
        origin: "fallback",
        rationale: `proposal failed: ${reason}`,
      });
    }
  }

  return { offspring, proposalCalls, firstExact };
}
