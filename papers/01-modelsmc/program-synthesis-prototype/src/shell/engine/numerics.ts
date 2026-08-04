export interface NormalizedWeights {
  readonly weights: number[];
  readonly usedUniformFallback: boolean;
}

export function normalizeLogWeights(logWeights: readonly number[]): NormalizedWeights {
  if (logWeights.length === 0) {
    throw new Error("cannot normalize an empty weight vector");
  }

  const positiveInfinity = logWeights.reduce<number[]>(
    (indices, value, index) => (value === Number.POSITIVE_INFINITY ? [...indices, index] : indices),
    [],
  );
  if (positiveInfinity.length > 0) {
    const mass = 1 / positiveInfinity.length;
    const selected = new Set(positiveInfinity);
    return {
      weights: logWeights.map((_value, index) => (selected.has(index) ? mass : 0)),
      usedUniformFallback: false,
    };
  }

  const finite = logWeights.map((value) =>
    Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY,
  );
  const maximum = Math.max(...finite);
  if (maximum === Number.NEGATIVE_INFINITY) {
    return {
      weights: finite.map(() => 1 / finite.length),
      usedUniformFallback: true,
    };
  }

  const exponentials = finite.map((value) =>
    value === Number.NEGATIVE_INFINITY ? 0 : Math.exp(value - maximum),
  );
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  if (!(total > 0) || !Number.isFinite(total)) {
    return {
      weights: finite.map(() => 1 / finite.length),
      usedUniformFallback: true,
    };
  }
  return {
    weights: exponentials.map((value) => value / total),
    usedUniformFallback: false,
  };
}

export function effectiveSampleSize(weights: readonly number[]): number {
  if (weights.length === 0) throw new Error("ESS requires at least one weight");
  const squaredTotal = weights.reduce((sum, weight) => sum + weight * weight, 0);
  if (!(squaredTotal > 0) || !Number.isFinite(squaredTotal)) return 0;
  return 1 / squaredTotal;
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude < 0.0001 || magnitude >= 1_000_000) return value.toExponential(6);
  return Number(value.toPrecision(7)).toString();
}
