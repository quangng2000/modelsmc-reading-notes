import {
  examplesHaveSignature,
  valueType,
} from "../../core/language.verify.js";
import { ConfigurationError } from "./errors.js";
import { numberField, optionalString, requireArray, requireRecord } from "./json.js";
import type { ExperimentConfig } from "./types.js";
import { parseConstants, parseExample, parseSignature } from "./values.js";

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

  return {
    name: optionalString(root, "name", "unnamed-synthesis-task"),
    examples,
    inputType: valueType(first.input),
    outputType: valueType(first.output),
    integerConstants: parseConstants(root.integerConstants),
    particles: numberField(
      root,
      "particles",
      8,
      (value) => Number.isSafeInteger(value) && value >= 1 && value <= 1_024,
      "an integer from 1 through 1024",
    ),
    iterations: numberField(
      root,
      "iterations",
      6,
      (value) => Number.isSafeInteger(value) && value >= 1 && value <= 10_000,
      "an integer from 1 through 10000",
    ),
    cloneProbability: numberField(
      root,
      "cloneProbability",
      0.35,
      (value) => value >= 0 && value <= 1,
      "between 0 and 1",
    ),
    essThreshold: numberField(
      root,
      "essThreshold",
      0.6,
      (value) => value > 0 && value <= 1,
      "a relative threshold in (0, 1]",
    ),
    seed: numberField(
      root,
      "seed",
      7,
      (value) => Number.isSafeInteger(value),
      "a safe integer",
    ),
    lossScale: numberField(root, "lossScale", 2, (value) => value > 0, "greater than zero"),
    costScale: numberField(root, "costScale", 0.15, (value) => value >= 0, "zero or greater"),
    lossCap: numberField(
      root,
      "lossCap",
      1_000_000,
      (value) => Number.isSafeInteger(value) && value >= 1,
      "a positive safe integer",
    ),
    maxCost: numberField(
      root,
      "maxCost",
      20,
      (value) => Number.isSafeInteger(value) && value >= 1,
      "a positive safe integer",
    ),
    maxDepth: numberField(
      root,
      "maxDepth",
      10,
      (value) => Number.isSafeInteger(value) && value >= 1 && value <= 64,
      "an integer from 1 through 64",
    ),
    maxNodes: numberField(
      root,
      "maxNodes",
      127,
      (value) => Number.isSafeInteger(value) && value >= 1 && value <= 10_000,
      "an integer from 1 through 10000",
    ),
  };
}
