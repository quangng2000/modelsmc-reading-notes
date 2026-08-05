import {
  boolCons,
  boolListValue,
  boolNil,
  boolValue,
  intCons,
  intListValue,
  intNil,
  intValue,
  type BoolList,
  type Example,
  type IntList,
  type RuntimeValue,
  type StaticType,
} from "../../core/language.verify.js";
import { ConfigurationError } from "./errors.js";
import { assertExactKeys, requireArray, requireRecord } from "./json.js";

export interface ParsedSignature {
  readonly inputType: StaticType;
  readonly outputType: StaticType;
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

export function parseSignature(value: unknown): ParsedSignature | undefined {
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

export function parseExample(
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

export function parseConstants(value: unknown): bigint[] {
  const raw =
    value === undefined
      ? ["-2", "-1", "0", "1", "2"]
      : requireArray(value, "integerConstants");
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
