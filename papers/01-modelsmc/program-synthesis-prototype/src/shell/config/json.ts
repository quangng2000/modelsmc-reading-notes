import { ConfigurationError } from "./errors.js";

export type JsonRecord = Record<string, unknown>;

export function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function requireRecord(value: unknown, path: string): JsonRecord {
  if (!isRecord(value)) throw new ConfigurationError(`${path} must be an object`);
  return value;
}

export function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new ConfigurationError(`${path} must be an array`);
  return value;
}

export function assertExactKeys(
  record: JsonRecord,
  expected: readonly string[],
  path: string,
): void {
  const actual = Object.keys(record).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw new ConfigurationError(
      `${path} must contain exactly: ${wanted.join(", ")}; received: ${actual.join(", ") || "(none)"}`,
    );
  }
}

export function optionalString(record: JsonRecord, key: string, fallback: string): string {
  const value = record[key];
  if (value === undefined) return fallback;
  if (typeof value !== "string" || value.trim() === "") {
    throw new ConfigurationError(`${key} must be a nonempty string`);
  }
  return value;
}

export function numberField(
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
