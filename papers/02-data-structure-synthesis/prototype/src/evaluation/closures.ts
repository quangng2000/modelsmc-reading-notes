import type { ClosureValue } from "./types.js";

const EVALUATOR_CLOSURES = new WeakSet<object>();

export function registerClosure(closure: ClosureValue): void {
  EVALUATOR_CLOSURES.add(closure);
}

export function isClosure(value: unknown): value is ClosureValue {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }

  const candidate = value as Partial<ClosureValue>;
  return candidate.kind === "closure" && EVALUATOR_CLOSURES.has(value);
}
