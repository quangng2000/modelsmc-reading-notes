import type { RandomSource } from "./random.js";

export function systematicResample(
  weights: readonly number[],
  random: RandomSource,
): number[] {
  if (weights.length === 0) throw new Error("resampling requires at least one weight");
  if (weights.some((weight) => !Number.isFinite(weight) || weight < 0)) {
    throw new Error("resampling weights must be finite and nonnegative");
  }
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  if (!(total > 0)) throw new Error("resampling weights must contain positive mass");

  const normalized = weights.map((weight) => weight / total);
  const count = normalized.length;
  const start = random.next() / count;
  const ancestors: number[] = [];
  let ancestor = 0;
  let cumulative = normalized[0]!;

  for (let slot = 0; slot < count; slot += 1) {
    const threshold = start + slot / count;
    while (ancestor < count - 1 && threshold > cumulative) {
      ancestor += 1;
      cumulative += normalized[ancestor]!;
    }
    ancestors.push(ancestor);
  }
  return ancestors;
}
