import {
  DEFAULT_CONSTANTS,
  DEFAULT_MAX_COST,
  DEFAULT_VARIABLES,
} from "./constants.js";
import type {
  PrimitiveTypeName,
  SearchOptions,
  TypedVariableOption,
  VariableOption,
} from "./types.js";

export function validateSearchOptions(options: SearchOptions = {}): void {
  validateMaxCost(options.maxCost ?? DEFAULT_MAX_COST);
  validateIntegers(options.constants ?? DEFAULT_CONSTANTS, "constants");
  validateStrings(options.stringConstants ?? [], "stringConstants");
  validateVariables(options.variables ?? DEFAULT_VARIABLES);
  validateTargetType(options.targetType);
}

export function validateIntegers(
  values: readonly number[],
  label: string,
): void {
  if (!values.every(Number.isSafeInteger)) {
    throw new Error(`${label} must contain only safe integers.`);
  }
}

export function validateStrings(
  values: readonly string[],
  label: string,
): void {
  if (!values.every((value) => typeof value === "string")) {
    throw new Error(`${label} must contain only strings.`);
  }
}

export function normalizeVariables(
  variables: readonly VariableOption[],
): readonly TypedVariableOption[] {
  if (variables.length === 0) {
    throw new Error("variables must contain at least one name.");
  }

  const normalized = variables.map((variable): TypedVariableOption => {
    if (typeof variable === "string") {
      if (variable.length === 0) {
        throw new Error("variables must contain only nonempty names.");
      }
      return { name: variable, type: "int" };
    }

    if (
      typeof variable !== "object" ||
      variable === null ||
      typeof variable.name !== "string" ||
      variable.name.length === 0 ||
      !isPrimitiveTypeName(variable.type)
    ) {
      throw new Error(
        "typed variables must have a nonempty name and an int, bool, or string type.",
      );
    }
    return variable;
  });

  if (new Set(normalized.map(({ name }) => name)).size !== normalized.length) {
    throw new Error("variables must be distinct.");
  }
  return normalized;
}

function validateMaxCost(maxCost: number): void {
  if (!Number.isSafeInteger(maxCost) || maxCost < 0) {
    throw new Error("maxCost must be a nonnegative safe integer.");
  }
}

function validateVariables(variables: readonly VariableOption[]): void {
  normalizeVariables(variables);
}

function validateTargetType(targetType: PrimitiveTypeName | undefined): void {
  if (targetType !== undefined && !isPrimitiveTypeName(targetType)) {
    throw new Error('targetType must be "int", "bool", or "string".');
  }
}

function isPrimitiveTypeName(value: unknown): value is PrimitiveTypeName {
  return value === "int" || value === "bool" || value === "string";
}
