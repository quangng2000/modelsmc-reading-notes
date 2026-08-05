import { ConfigurationError } from "./errors.js";
import type { ConfigOverrides, ExperimentConfig } from "./types.js";

export function withConfigOverrides(
  config: ExperimentConfig,
  overrides: ConfigOverrides,
): ExperimentConfig {
  const merged: ExperimentConfig = {
    ...config,
    ...(overrides.particles === undefined ? {} : { particles: overrides.particles }),
    ...(overrides.iterations === undefined ? {} : { iterations: overrides.iterations }),
    ...(overrides.cloneProbability === undefined
      ? {}
      : { cloneProbability: overrides.cloneProbability }),
    ...(overrides.essThreshold === undefined
      ? {}
      : { essThreshold: overrides.essThreshold }),
    ...(overrides.seed === undefined ? {} : { seed: overrides.seed }),
  };

  if (!Number.isSafeInteger(merged.particles) || merged.particles < 1) {
    throw new ConfigurationError("particles override must be a positive integer");
  }
  if (!Number.isSafeInteger(merged.iterations) || merged.iterations < 1) {
    throw new ConfigurationError("iterations override must be a positive integer");
  }
  if (merged.cloneProbability < 0 || merged.cloneProbability > 1) {
    throw new ConfigurationError("clone probability override must be between 0 and 1");
  }
  if (merged.essThreshold <= 0 || merged.essThreshold > 1) {
    throw new ConfigurationError("ESS threshold override must be in (0, 1]");
  }
  if (!Number.isSafeInteger(merged.seed)) {
    throw new ConfigurationError("seed override must be a safe integer");
  }
  return merged;
}
