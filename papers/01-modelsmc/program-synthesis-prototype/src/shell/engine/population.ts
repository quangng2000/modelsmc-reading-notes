import type { Program } from "../../core/language.verify.js";
import type { ExperimentConfig } from "../config/index.js";
import type { ScoringOptions } from "../scoring/index.js";
import type { Particle } from "./types.js";

export function scoringOptions(config: ExperimentConfig): ScoringOptions {
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

export function initialExpressions(config: ExperimentConfig): Program[] {
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

export function bestParticle(particles: readonly Particle[]): Particle {
  if (particles.length === 0) throw new Error("particle population is empty");
  return particles.reduce((best, candidate) =>
    candidate.score.logTarget > best.score.logTarget ? candidate : best,
  );
}

export function betterChampion(candidate: Particle, current: Particle): boolean {
  if (candidate.score.exactProgram !== current.score.exactProgram) {
    return candidate.score.exactProgram;
  }
  return candidate.score.logTarget > current.score.logTarget;
}

export function bestChampion(particles: readonly Particle[]): Particle {
  if (particles.length === 0) throw new Error("particle population is empty");
  return particles.reduce((best, candidate) =>
    betterChampion(candidate, best) ? candidate : best,
  );
}

export function particleLineage(
  particle: Particle,
  archive: ReadonlyMap<number, Particle>,
): Particle[] {
  const reversed: Particle[] = [];
  const visited = new Set<number>();
  let current: Particle | undefined = particle;
  while (current !== undefined && !visited.has(current.id)) {
    reversed.push(current);
    visited.add(current.id);
    current = current.parentId === null ? undefined : archive.get(current.parentId);
  }
  return reversed.reverse();
}
