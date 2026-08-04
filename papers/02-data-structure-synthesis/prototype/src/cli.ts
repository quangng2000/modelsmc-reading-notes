#!/usr/bin/env node
import { readFileSync } from "node:fs";

import {
  BOOL,
  INT,
  STRING,
  listOf,
  renderExpression,
  renderType,
  type ObjectType,
  type PrimitiveType,
  type PrimitiveValue,
} from "./ast.js";
import { reportRequest, reportSynthesisEvent } from "./cli/trace.js";
import type { SearchOptions } from "./enumeration/index.js";
import { resolveSignature } from "./synthesis/signature.js";
import {
  synthesizeProgram,
  type FamilyReport,
  type IOExample,
  type SynthesisOptions,
  type SynthesisOutcome,
  type SynthesisSignature,
} from "./synthesis/index.js";

const USAGE =
  "usage: lambda2-synth [--trace] <examples.json> | lambda2-synth [--trace] -e '<json>'";

const EXIT_SYNTHESIZED = 0;
const EXIT_NO_PROGRAM = 1;
const EXIT_INVALID_INPUT = 2;

interface CliRequest {
  readonly examples: readonly IOExample[];
  readonly signature: SynthesisSignature;
  readonly options: SearchOptions;
}

interface CliArguments {
  readonly inputArguments: readonly string[];
  readonly trace: boolean;
}

function runCli(argv: readonly string[]): number {
  const cliArguments = parseCliArguments(argv);
  if (cliArguments === undefined) {
    return EXIT_INVALID_INPUT;
  }

  const text = readInputText(cliArguments.inputArguments);
  if (text === undefined) {
    return EXIT_INVALID_INPUT;
  }

  let request: CliRequest;
  try {
    request = parseRequest(text);
  } catch (error) {
    process.stderr.write(`error: ${errorMessage(error)}\n`);
    return EXIT_INVALID_INPUT;
  }

  const synthesisOptions: SynthesisOptions = {
    ...request.options,
    inputType: request.signature.inputType,
    outputType: request.signature.outputType,
    ...(cliArguments.trace ? { onEvent: reportSynthesisEvent } : {}),
  };

  if (cliArguments.trace) {
    reportRequest(request.examples, request.signature);
  }

  return reportOutcome(synthesizeProgram(request.examples, synthesisOptions));
}

function parseCliArguments(argv: readonly string[]): CliArguments | undefined {
  const inputArguments: string[] = [];
  let trace = false;

  for (const argument of argv) {
    if (argument === "--trace" || argument === "-t") {
      if (trace) {
        process.stderr.write(`${USAGE}\n`);
        return undefined;
      }
      trace = true;
      continue;
    }
    inputArguments.push(argument);
  }

  return { inputArguments, trace };
}

function readInputText(argv: readonly string[]): string | undefined {
  if (argv.length === 2 && argv[0] === "-e" && argv[1] !== undefined) {
    return argv[1];
  }

  const path = argv[0];
  if (argv.length !== 1 || path === undefined || path === "-e") {
    process.stderr.write(`${USAGE}\n`);
    return undefined;
  }

  try {
    return readFileSync(path, "utf8");
  } catch (error) {
    process.stderr.write(
      `error: cannot read ${path}: ${errorMessage(error)}\n`,
    );
    return undefined;
  }
}

function parseRequest(text: string): CliRequest {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("input is not valid JSON");
  }

  if (!isRecord(parsed)) {
    throw new Error('input must be a JSON object with an "examples" array');
  }

  const inputTypeText = parsed["inputType"];
  const outputTypeText = parsed["outputType"];
  if ((inputTypeText === undefined) !== (outputTypeText === undefined)) {
    throw new Error('"inputType" and "outputType" must be provided together');
  }

  if (inputTypeText === undefined || outputTypeText === undefined) {
    const examples = parseLegacyExamples(parsed["examples"]);
    return {
      examples,
      signature: resolveSignature(examples, {}),
      options: parseOptions(parsed),
    };
  }

  if (typeof inputTypeText !== "string" || typeof outputTypeText !== "string") {
    throw new Error('"inputType" and "outputType" must be type strings');
  }
  const declaredInput = parseType(inputTypeText);
  const declaredOutput = parseType(outputTypeText);
  const signature = resolveSignature([], {
    inputType: declaredInput,
    outputType: declaredOutput,
  });
  const examples = parseTypedExamples(parsed["examples"], signature);

  return {
    examples,
    signature: resolveSignature(examples, {
      inputType: signature.inputType,
      outputType: signature.outputType,
    }),
    options: parseOptions(parsed),
  };
}

function parseLegacyExamples(raw: unknown): readonly IOExample[] {
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new Error('"examples" must be a nonempty array');
  }

  const examples = raw.map((example, index) =>
    parseLegacyExample(example, index),
  );
  const scalarOutputs = examples.filter(
    (example) => typeof example.output === "number",
  ).length;
  if (scalarOutputs !== 0 && scalarOutputs !== examples.length) {
    throw new Error(
      "example outputs must be all lists or all integers, not a mix",
    );
  }
  return examples;
}

function parseLegacyExample(raw: unknown, index: number): IOExample {
  if (!isRecord(raw)) {
    throw new Error(`example ${index + 1} must be an object`);
  }

  const input = raw["input"];
  if (!isSafeIntegerArray(input)) {
    throw new Error(
      `example ${index + 1} input must be an array of safe integers`,
    );
  }

  const output = raw["output"];
  if (isSafeInteger(output) || isSafeIntegerArray(output)) {
    return { input, output };
  }
  throw new Error(
    `example ${index + 1} output must be a safe integer or an array of safe integers`,
  );
}

function parseTypedExamples(
  raw: unknown,
  signature: SynthesisSignature,
): readonly IOExample[] {
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new Error('"examples" must be a nonempty array');
  }
  return raw.map((example, index) => {
    if (!isRecord(example)) {
      throw new Error(`example ${index + 1} must be an object`);
    }
    const input = parsePrimitiveList(
      example["input"],
      signature.inputType.element,
      `example ${index + 1} input`,
    );
    const output =
      signature.outputType.kind === "list"
        ? parsePrimitiveList(
            example["output"],
            signature.outputType.element,
            `example ${index + 1} output`,
          )
        : parsePrimitive(
            example["output"],
            signature.outputType,
            `example ${index + 1} output`,
          );
    return { input, output };
  });
}

function parsePrimitiveList(
  raw: unknown,
  type: PrimitiveType,
  label: string,
): readonly PrimitiveValue[] {
  if (!Array.isArray(raw)) {
    throw new Error(`${label} must match list<${renderType(type)}>`);
  }
  return raw.map((value, index) =>
    parsePrimitive(value, type, `${label}[${index}]`),
  );
}

function parsePrimitive(
  raw: unknown,
  type: PrimitiveType,
  label: string,
): PrimitiveValue {
  switch (type.kind) {
    case "int":
      if (isSafeInteger(raw)) {
        return raw;
      }
      break;
    case "bool":
      if (typeof raw === "boolean") {
        return raw;
      }
      break;
    case "string":
      if (typeof raw === "string") {
        return raw;
      }
      break;
  }
  throw new Error(`${label} must match ${renderType(type)}`);
}

function parseType(text: string): ObjectType {
  const source = text.replace(/\s+/g, "");
  switch (source) {
    case "int":
      return INT;
    case "bool":
      return BOOL;
    case "string":
      return STRING;
  }
  if (source.startsWith("list<") && source.endsWith(">")) {
    const elementText = source.slice(5, -1);
    if (elementText.length === 0) {
      throw new Error(`invalid type: ${JSON.stringify(text)}`);
    }
    return listOf(parseType(elementText));
  }
  throw new Error(`invalid type: ${JSON.stringify(text)}`);
}

function parseOptions(
  parsed: Readonly<Record<string, unknown>>,
): SearchOptions {
  let options: SearchOptions = {};

  const maxCost = parsed["maxCost"];
  if (maxCost !== undefined) {
    if (!isSafeInteger(maxCost) || maxCost < 0) {
      throw new Error('"maxCost" must be a nonnegative safe integer');
    }
    options = { ...options, maxCost };
  }

  const constants = parsed["constants"];
  if (constants !== undefined) {
    if (!isSafeIntegerArray(constants)) {
      throw new Error('"constants" must be an array of safe integers');
    }
    options = { ...options, constants };
  }

  const stringConstants = parsed["stringConstants"];
  if (stringConstants !== undefined) {
    if (!isStringArray(stringConstants)) {
      throw new Error('"stringConstants" must be an array of strings');
    }
    options = { ...options, stringConstants };
  }

  return options;
}

function reportOutcome(outcome: SynthesisOutcome): number {
  switch (outcome.kind) {
    case "synthesized": {
      writeLines([
        `family:  ${outcome.family}`,
        `program: ${renderExpression(outcome.program)}`,
        `cost:    ${outcome.cost}`,
        `tested:  ${outcome.candidatesTested} candidates`,
        "deduction:",
        ...outcome.familyReports.map(renderFamilyReport),
      ]);
      return EXIT_SYNTHESIZED;
    }
    case "refuted":
    case "not-found": {
      writeLines([
        `${outcome.kind}: ${outcome.reason}`,
        "deduction:",
        ...outcome.familyReports.map(renderFamilyReport),
      ]);
      return EXIT_NO_PROGRAM;
    }
  }
}

function renderFamilyReport(report: FamilyReport): string {
  return report.status === "viable"
    ? `  ${report.family}: viable`
    : `  ${report.family}: refuted — ${report.reason ?? "no reason recorded"}`;
}

function writeLines(lines: readonly string[]): void {
  process.stdout.write(`${lines.join("\n")}\n`);
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSafeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value);
}

function isSafeIntegerArray(value: unknown): value is readonly number[] {
  return Array.isArray(value) && value.every(isSafeInteger);
}

function isStringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

process.exitCode = runCli(process.argv.slice(2));
