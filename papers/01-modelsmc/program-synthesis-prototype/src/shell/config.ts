import {
  boolCons,
  boolListValue,
  boolNil,
  boolValue,
  examplesHaveSignature,
  intCons,
  intListValue,
  intNil,
  intValue,
  valueType,
  type BoolList,
  type Example,
  type IntList,
  type RuntimeValue,
  type StaticType,
} from "../core/language.verify.js";

export class ConfigurationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigurationError";
  }
}

export interface ExperimentConfig {
  readonly name: string;
  readonly examples: readonly Example[];
  readonly inputType: StaticType;
  readonly outputType: StaticType;
  readonly integerConstants: readonly bigint[];
  readonly particles: number;
  readonly iterations: number;
  readonly cloneProbability: number;
  readonly essThreshold: number;
  readonly seed: number;
  readonly lossScale: number;
  readonly costScale: number;
  readonly lossCap: number;
  readonly maxCost: number;
  readonly maxDepth: number;
  readonly maxNodes: number;
}

export interface ConfigOverrides {
  readonly particles?: number;
  readonly iterations?: number;
  readonly cloneProbability?: number;
  readonly essThreshold?: number;
  readonly seed?: number;
}

type JsonRecord = Record<string, unknown>;

interface ParsedSignature {
  readonly inputType: StaticType;
  readonly outputType: StaticType;
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, path: string): JsonRecord {
  if (!isRecord(value)) throw new ConfigurationError(`${path} must be an object`);
  return value;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new ConfigurationError(`${path} must be an array`);
  return value;
}

function assertExactKeys(record: JsonRecord, expected: readonly string[], path: string): void {
  const actual = Object.keys(record).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw new ConfigurationError(
      `${path} must contain exactly: ${wanted.join(", ")}; received: ${actual.join(", ") || "(none)"}`,
    );
  }
}

function optionalString(record: JsonRecord, key: string, fallback: string): string {
  const value = record[key];
  if (value === undefined) return fallback;
  if (typeof value !== "string" || value.trim() === "") {
    throw new ConfigurationError(`${key} must be a nonempty string`);
  }
  return value;
}

function numberField(
  record: JsonRecord,
  key: string,
  fallback: number,
  predicate: (value: number) => boolean,
  expectation: string,
): number {
  const raw = record[key];
  if (raw === undefined) return fallback;
  if (typeof raw !== "number" || !Number.isFinite(raw) || !predicate(raw)) {
    throw new ConfigurationError(`${key} must be ${expectation}`);
  }
  return raw;
}

const DECIMAL_INTEGER = /^-?(?:0|[1-9][0-9]*)$/;

export function parseDecimalBigInt(value: unknown, path: string): bigint {
  if (typeof value !== "string" || !DECIMAL_INTEGER.test(value)) {
    throw new ConfigurationError(
      `${path} must be a decimal integer string such as "-2", "0", or "17"`,
    );
  }
  try {
    return BigInt(value);
  } catch {
    throw new ConfigurationError(`${path} is not a valid integer`);
  }
}

function parseLegacyRuntimeValue(value: unknown, path: string): RuntimeValue {
  if (Array.isArray(value)) {
    throw new ConfigurationError(
      `${path} is a list, so configuration.signature must explicitly declare its element type`,
    );
  }
  if (typeof value === "boolean") return boolValue(value);
  return intValue(parseDecimalBigInt(value, path));
}

function parseSignatureType(value: unknown, path: string): StaticType {
  if (value === "Int") return "IntType";
  if (value === "Bool") return "BoolType";
  if (value === "List<Int>") return "IntListType";
  if (value === "List<Bool>") return "BoolListType";
  throw new ConfigurationError(
    `${path} must be one of "Int", "Bool", "List<Int>", or "List<Bool>"`,
  );
}

function parseSignature(value: unknown): ParsedSignature | undefined {
  if (value === undefined) return undefined;
  const record = requireRecord(value, "signature");
  assertExactKeys(record, ["input", "output"], "signature");
  return {
    inputType: parseSignatureType(record.input, "signature.input"),
    outputType: parseSignatureType(record.output, "signature.output"),
  };
}

function parseIntList(value: unknown, path: string): RuntimeValue {
  const items = requireArray(value, path);
  let result: IntList = intNil();
  for (let index = items.length - 1; index >= 0; index -= 1) {
    result = intCons(parseDecimalBigInt(items[index], `${path}[${index}]`), result);
  }
  return intListValue(result);
}

function parseBoolList(value: unknown, path: string): RuntimeValue {
  const items = requireArray(value, path);
  let result: BoolList = boolNil();
  for (let index = items.length - 1; index >= 0; index -= 1) {
    const item = items[index];
    if (typeof item !== "boolean") {
      throw new ConfigurationError(`${path}[${index}] must be a Boolean`);
    }
    result = boolCons(item, result);
  }
  return boolListValue(result);
}

function parseRuntimeValueForType(
  value: unknown,
  expectedType: StaticType,
  path: string,
): RuntimeValue {
  if (expectedType === "IntType") return intValue(parseDecimalBigInt(value, path));
  if (expectedType === "BoolType") {
    if (typeof value !== "boolean") throw new ConfigurationError(`${path} must be a Boolean`);
    return boolValue(value);
  }
  if (expectedType === "IntListType") return parseIntList(value, path);
  return parseBoolList(value, path);
}

function parseExample(
  value: unknown,
  index: number,
  signature: ParsedSignature | undefined,
): Example {
  const record = requireRecord(value, `examples[${index}]`);
  if (!("input" in record)) {
    throw new ConfigurationError(`examples[${index}].input is required`);
  }
  if (!("output" in record)) {
    throw new ConfigurationError(`examples[${index}].output is required`);
  }
  return {
    input:
      signature === undefined
        ? parseLegacyRuntimeValue(record.input, `examples[${index}].input`)
        : parseRuntimeValueForType(
            record.input,
            signature.inputType,
            `examples[${index}].input`,
          ),
    output:
      signature === undefined
        ? parseLegacyRuntimeValue(record.output, `examples[${index}].output`)
        : parseRuntimeValueForType(
            record.output,
            signature.outputType,
            `examples[${index}].output`,
          ),
  };
}

function parseConstants(value: unknown): bigint[] {
  const raw = value === undefined ? ["-2", "-1", "0", "1", "2"] : requireArray(value, "integerConstants");
  if (raw.length === 0) {
    throw new ConfigurationError("integerConstants must contain at least one value");
  }
  const result: bigint[] = [];
  const seen = new Set<string>();
  for (let index = 0; index < raw.length; index += 1) {
    const parsed = parseDecimalBigInt(raw[index], `integerConstants[${index}]`);
    const canonical = parsed.toString();
    if (!seen.has(canonical)) {
      seen.add(canonical);
      result.push(parsed);
    }
  }
  return result;
}

export function parseExperimentConfig(text: string): ExperimentConfig {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text) as unknown;
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new ConfigurationError(`configuration is not valid JSON: ${detail}`);
  }

  const root = requireRecord(parsed, "configuration");
  const signature = parseSignature(root.signature);
  const rawExamples = requireArray(root.examples, "examples");
  const examples = rawExamples.map((value, index) => parseExample(value, index, signature));
  if (!examplesHaveSignature(examples)) {
    throw new ConfigurationError(
      "examples must be nonempty and share one input type and one output type",
    );
  }

  const first = examples[0]!;
  if (
    signature !== undefined &&
    (valueType(first.input) !== signature.inputType ||
      valueType(first.output) !== signature.outputType)
  ) {
    throw new ConfigurationError("examples do not match the declared signature");
  }
  const particles = numberField(
    root,
    "particles",
    8,
    (value) => Number.isSafeInteger(value) && value >= 1 && value <= 1_024,
    "an integer from 1 through 1024",
  );
  const iterations = numberField(
    root,
    "iterations",
    6,
    (value) => Number.isSafeInteger(value) && value >= 1 && value <= 10_000,
    "an integer from 1 through 10000",
  );
  const cloneProbability = numberField(
    root,
    "cloneProbability",
    0.35,
    (value) => value >= 0 && value <= 1,
    "between 0 and 1",
  );
  const essThreshold = numberField(
    root,
    "essThreshold",
    0.6,
    (value) => value > 0 && value <= 1,
    "a relative threshold in (0, 1]",
  );
  const seed = numberField(
    root,
    "seed",
    7,
    (value) => Number.isSafeInteger(value),
    "a safe integer",
  );
  const lossScale = numberField(
    root,
    "lossScale",
    2,
    (value) => value > 0,
    "greater than zero",
  );
  const costScale = numberField(
    root,
    "costScale",
    0.15,
    (value) => value >= 0,
    "zero or greater",
  );
  const lossCap = numberField(
    root,
    "lossCap",
    1_000_000,
    (value) => Number.isSafeInteger(value) && value >= 1,
    "a positive safe integer",
  );
  const maxCost = numberField(
    root,
    "maxCost",
    20,
    (value) => Number.isSafeInteger(value) && value >= 1,
    "a positive safe integer",
  );
  const maxDepth = numberField(
    root,
    "maxDepth",
    10,
    (value) => Number.isSafeInteger(value) && value >= 1 && value <= 64,
    "an integer from 1 through 64",
  );
  const maxNodes = numberField(
    root,
    "maxNodes",
    127,
    (value) => Number.isSafeInteger(value) && value >= 1 && value <= 10_000,
    "an integer from 1 through 10000",
  );

  return {
    name: optionalString(root, "name", "unnamed-synthesis-task"),
    examples,
    inputType: valueType(first.input),
    outputType: valueType(first.output),
    integerConstants: parseConstants(root.integerConstants),
    particles,
    iterations,
    cloneProbability,
    essThreshold,
    seed,
    lossScale,
    costScale,
    lossCap,
    maxCost,
    maxDepth,
    maxNodes,
  };
}

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
