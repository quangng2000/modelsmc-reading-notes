import type { EvaluationEnvironment, ValueBinding } from "./types.js";
import { valueMatchesType } from "./values.js";

export function findBinding(
  environment: EvaluationEnvironment,
  name: string,
): ValueBinding | undefined {
  for (let index = environment.length - 1; index >= 0; index -= 1) {
    const binding = environment[index];
    if (binding !== undefined && binding.name === name) {
      return binding;
    }
  }
  return undefined;
}

export function validateEnvironment(
  environment: EvaluationEnvironment,
): void {
  for (const binding of environment) {
    if (!valueMatchesType(binding.value, binding.type)) {
      throw new TypeError(
        `Environment value for ${binding.name} does not match its declared type.`,
      );
    }
  }
}
