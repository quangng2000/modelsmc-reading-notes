import type { RandomSource } from "../engine/random.js";

export function logSumExp(values: readonly number[]): number {
  if (values.length === 0) return Number.NEGATIVE_INFINITY;
  const maximum = Math.max(...values);
  if (maximum === Number.NEGATIVE_INFINITY) return maximum;
  if (maximum === Number.POSITIVE_INFINITY) return maximum;
  const total = values.reduce((sum, value) => sum + Math.exp(value - maximum), 0);
  return maximum + Math.log(total);
}

export function cumulativeProbabilities(probabilities: readonly number[]): number[] {
  if (probabilities.length === 0) throw new Error("a categorical distribution cannot be empty");
  const cumulative: number[] = [];
  let total = 0;
  for (const probability of probabilities) {
    if (!Number.isFinite(probability) || probability < 0) {
      throw new Error("categorical probabilities must be finite and nonnegative");
    }
    total += probability;
    cumulative.push(total);
  }
  if (!(total > 0)) throw new Error("categorical probabilities must contain positive mass");
  return cumulative.map((value) => value / total);
}

export function sampleCategorical(
  cumulative: readonly number[],
  random: RandomSource,
): number {
  if (cumulative.length === 0) throw new Error("a categorical distribution cannot be empty");
  const draw = random.next();
  let low = 0;
  let high = cumulative.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (draw < cumulative[middle]!) high = middle;
    else low = middle + 1;
  }
  return low;
}

export function totalVariation(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length) throw new Error("TV distance requires equal-length vectors");
  return 0.5 * left.reduce((sum, value, index) => sum + Math.abs(value - right[index]!), 0);
}
