import { exprToJsonValue, renderExpr, renderType, renderValue } from "../ast/render.js";
import { formatNumber } from "./numerics.js";
import type { TraceSink } from "./trace.js";
import type { Particle } from "./types.js";

export function traceParticleScore(
  trace: TraceSink,
  particle: Particle,
  phase: string,
): void {
  trace.emit(
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
    trace.emit(
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
