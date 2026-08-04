import {
  acceptProgram,
  inferType,
  type Program,
} from "../core/language.verify.js";
import type { ExperimentConfig } from "./config.js";
import { deriveSynthesisTrace } from "./deduction.js";
import { effectiveSampleSize, formatNumber, normalizeLogWeights } from "./numerics.js";
import type { Proposer } from "./proposal.js";
import { SeededRandom } from "./random.js";
import { renderExpr, renderType, renderValue, exprToJsonValue } from "./render.js";
import { systematicResample } from "./resampling.js";
import {
  scoreProgram,
  type ScoringOptions,
  type ValidScore,
} from "./scoring.js";
import { NullTraceSink, type TraceSink } from "./trace.js";

export interface Particle {
  readonly id: number;
  readonly parentId: number | null;
  readonly expression: Program;
  readonly score: ValidScore;
  readonly weight: number;
  readonly origin: "initial" | "clone" | "catalog" | "ollama" | "fallback";
  readonly rationale: string;
}

export interface IterationSummary {
  readonly iteration: number;
  readonly essBefore: number;
  readonly relativeEssBefore: number;
  readonly resampled: boolean;
  readonly ancestors: readonly number[];
  readonly bestParticleId: number;
  readonly bestExpression: Program;
  readonly bestWeight: number;
}

export interface SynthesisResult {
  readonly particles: readonly Particle[];
  readonly iterations: readonly IterationSummary[];
  readonly best: Particle;
  readonly exact: boolean;
  readonly proposalCalls: number;
}

export interface EngineOptions {
  readonly config: ExperimentConfig;
  readonly proposer: Proposer;
  readonly trace?: TraceSink;
}

function scoringOptions(config: ExperimentConfig): ScoringOptions {
  return {
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    lossScale: config.lossScale,
    costScale: config.costScale,
    lossCap: config.lossCap,
    maxCost: config.maxCost,
  };
}

function initialExpressions(config: ExperimentConfig): Program[] {
  const candidates: Program[] = [];
  if (config.inputType === config.outputType) {
    candidates.push({ kind: "ExpressionProgram", body: { kind: "Input" } });
  }
  if (config.outputType === "IntType") {
    for (const value of config.integerConstants) {
      candidates.push({
        kind: "ExpressionProgram",
        body: { kind: "IntLiteral", intValue: value },
      });
    }
  } else if (config.outputType === "BoolType") {
    candidates.push(
      {
        kind: "ExpressionProgram",
        body: { kind: "BoolLiteral", boolValue: false },
      },
      {
        kind: "ExpressionProgram",
        body: { kind: "BoolLiteral", boolValue: true },
      },
    );
  } else if (config.outputType === "IntListType") {
    candidates.push({
      kind: "ExpressionProgram",
      body: { kind: "EmptyIntList" },
    });
  } else {
    candidates.push({
      kind: "ExpressionProgram",
      body: { kind: "EmptyBoolList" },
    });
  }
  return candidates;
}

function bestParticle(particles: readonly Particle[]): Particle {
  if (particles.length === 0) throw new Error("particle population is empty");
  return particles.reduce((best, candidate) =>
    candidate.score.logTarget > best.score.logTarget ? candidate : best,
  );
}

function betterChampion(candidate: Particle, current: Particle): boolean {
  if (candidate.score.exactProgram !== current.score.exactProgram) {
    return candidate.score.exactProgram;
  }
  return candidate.score.logTarget > current.score.logTarget;
}

function bestChampion(particles: readonly Particle[]): Particle {
  if (particles.length === 0) throw new Error("particle population is empty");
  return particles.reduce((best, candidate) =>
    betterChampion(candidate, best) ? candidate : best,
  );
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

  private traceScore(particle: Particle, phase: string): void {
    this.trace.emit(
      "particle.scored",
      `${phase} particle=${particle.id}: ${renderExpr(particle.expression)}; type=${renderType(particle.score.inferredType)} cost=${particle.score.cost} loss=${formatNumber(particle.score.totalLoss)} log-target=${formatNumber(particle.score.logTarget)} weight=${formatNumber(particle.weight)}`,
      {
        phase,
        particleId: particle.id,
        parentId: particle.parentId,
        expression: exprToJsonValue(particle.expression),
        renderedExpression: renderExpr(particle.expression),
        inferredType: particle.score.inferredType,
        cost: particle.score.cost,
        totalLoss: particle.score.totalLoss,
        logTarget: particle.score.logTarget,
        weight: particle.weight,
        exactProgram: particle.score.exactProgram,
        origin: particle.origin,
      },
    );
    particle.score.evaluations.forEach((evaluation, index) => {
      this.trace.emit(
        "example.evaluated",
        `particle=${particle.id} example=${index + 1}: ${renderValue(evaluation.input)} -> predicted ${renderValue(evaluation.predicted)}, expected ${renderValue(evaluation.expected)}, loss=${formatNumber(evaluation.loss)}${evaluation.exact ? " exact" : ""}`,
        {
          phase,
          particleId: particle.id,
          example: index + 1,
          input: renderValue(evaluation.input),
          predicted: renderValue(evaluation.predicted),
          expected: renderValue(evaluation.expected),
          loss: evaluation.loss,
          exact: evaluation.exact,
        },
      );
    });
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
    for (const particle of weighted) this.traceScore(particle, phase);
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
    let champion = bestChampion(particles);
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

      const offspring: Omit<Particle, "weight">[] = [];
      for (let slot = 0; slot < particles.length; slot += 1) {
        const ancestorIndex = ancestors[slot]!;
        const ancestor = particles[ancestorIndex]!;
        const draw = this.random.next();
        if (draw < this.config.cloneProbability) {
          const clone = {
            id: this.allocateId(),
            parentId: ancestor.id,
            expression: ancestor.expression,
            score: ancestor.score,
            origin: "clone" as const,
            rationale: "clone draw below alpha",
          };
          this.trace.emit(
            "proposal.cloned",
            `iteration ${iteration} slot=${slot}: draw=${formatNumber(draw)} < alpha=${formatNumber(this.config.cloneProbability)}; cloned particle ${ancestor.id}`,
            { iteration, slot, draw, alpha: this.config.cloneProbability, ancestorId: ancestor.id },
          );
          offspring.push(clone);
          continue;
        }

        const requestIndex = this.proposalCalls;
        this.proposalCalls += 1;
        this.trace.emit(
          "proposal.requested",
          `iteration ${iteration} slot=${slot}: draw=${formatNumber(draw)} >= alpha=${formatNumber(this.config.cloneProbability)}; asking ${this.proposer.name} to revise particle ${ancestor.id}`,
          {
            iteration,
            slot,
            draw,
            alpha: this.config.cloneProbability,
            ancestorId: ancestor.id,
            requestIndex,
          },
        );

        try {
          const proposal = await this.proposer.propose({
            requestIndex,
            inputType: this.config.inputType,
            outputType: this.config.outputType,
            examples: this.config.examples,
            integerConstants: this.config.integerConstants,
            maxDepth: this.config.maxDepth,
            maxNodes: this.config.maxNodes,
            maxCost: this.config.maxCost,
            ancestor: ancestor.expression,
            ancestorScore: ancestor.score,
          });
          const inferred = inferType(proposal.expression, this.config.inputType);
          this.trace.emit(
            "type.checked",
            `iteration ${iteration} slot=${slot}: type check for ${renderExpr(proposal.expression)} -> ${inferred.kind === "TypeOk" ? renderType(inferred.inferred) : "TypeError"}; expected ${renderType(this.config.outputType)}`,
            {
              iteration,
              slot,
              expression: exprToJsonValue(proposal.expression),
              result: inferred.kind,
              inferredType: inferred.kind === "TypeOk" ? inferred.inferred : null,
              expectedType: this.config.outputType,
            },
          );
          const score = scoreProgram(proposal.expression, scoringOptions(this.config));
          if (score.kind === "Rejected") {
            this.trace.emit(
              "proposal.rejected",
              `iteration ${iteration} slot=${slot}: proposed ${renderExpr(proposal.expression)}; ${score.reason}; retaining ancestor ${ancestor.id}`,
              {
                iteration,
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
              id: this.allocateId(),
              parentId: ancestor.id,
              expression: ancestor.expression,
              score: ancestor.score,
              origin: "fallback",
              rationale: `proposal rejected: ${score.reason}`,
            });
          } else {
            this.trace.emit(
              "proposal.accepted",
              `iteration ${iteration} slot=${slot}: ${proposal.source} proposed ${renderExpr(proposal.expression)}; inferred ${renderType(score.inferredType)}; accepted for scoring`,
              {
                iteration,
                slot,
                ancestorId: ancestor.id,
                expression: exprToJsonValue(proposal.expression),
                renderedExpression: renderExpr(proposal.expression),
                inferredType: score.inferredType,
                rationale: proposal.rationale,
                source: proposal.source,
              },
            );
            offspring.push({
              id: this.allocateId(),
              parentId: ancestor.id,
              expression: proposal.expression,
              score,
              origin: proposal.source,
              rationale: proposal.rationale,
            });
          }
        } catch (error) {
          const reason = error instanceof Error ? error.message : String(error);
          this.trace.emit(
            "proposal.failed",
            `iteration ${iteration} slot=${slot}: ${this.proposer.name} failed (${reason}); retaining ancestor ${ancestor.id}`,
            { iteration, slot, ancestorId: ancestor.id, reason, proposer: this.proposer.name },
          );
          offspring.push({
            id: this.allocateId(),
            parentId: ancestor.id,
            expression: ancestor.expression,
            score: ancestor.score,
            origin: "fallback",
            rationale: `proposal failed: ${reason}`,
          });
        }
      }

      // Score belongs to each current AST. Old particle weights are deliberately not
      // multiplied in again, which would count the same examples repeatedly.
      particles = this.normalizePopulation(offspring, `iteration-${iteration}`);
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
        `iteration ${iteration} complete: best particle=${best.id} ${renderExpr(best.expression)} loss=${formatNumber(best.score.totalLoss)} cost=${best.score.cost} weight=${formatNumber(best.weight)}`,
        {
          iteration,
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
    this.trace.emit(
      "run.completed",
      `finished: best-so-far=${renderExpr(best.expression)} loss=${formatNumber(best.score.totalLoss)} cost=${best.score.cost}; exact=${exact}`,
      {
        bestParticleId: best.id,
        bestExpression: exprToJsonValue(best.expression),
        bestLoss: best.score.totalLoss,
        bestCost: best.score.cost,
        bestWeight: best.weight,
        exact,
        proposalCalls: this.proposalCalls,
      },
    );
    return { particles, iterations: summaries, best, exact, proposalCalls: this.proposalCalls };
  }
}
