import type { Expression } from "../ast.js";
import { inferType, type TypeBinding } from "../typecheck.js";
import { validateEnvironment } from "./environment.js";
import { containsHole } from "./expression-inspection.js";
import { evaluateWellTyped } from "./runtime.js";
import { snapshotEnvironment } from "./snapshots.js";
import type { EvaluationEnvironment, Value } from "./types.js";

export function evaluateExpression(
  expression: Expression,
  environment: EvaluationEnvironment = [],
): Value {
  if (containsHole(expression)) {
    throw new Error("Cannot evaluate an expression with unresolved holes.");
  }

  const stableEnvironment = snapshotEnvironment(environment);
  validateEnvironment(stableEnvironment);
  const typeEnvironment: readonly TypeBinding[] = stableEnvironment;
  if (inferType(expression, typeEnvironment) === undefined) {
    throw new TypeError("Cannot evaluate an ill-typed expression.");
  }
  return evaluateWellTyped(expression, stableEnvironment);
}
