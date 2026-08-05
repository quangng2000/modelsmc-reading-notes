import {
  intValue,
  type Example,
  type Program,
} from "../../src/core/language.verify.js";
import {
  parseExperimentConfig,
  type ExperimentConfig,
} from "../../src/shell/config/index.js";

export const twicePlusOne: Program = {
  kind: "ExpressionProgram",
  body: {
    kind: "Add",
    left: {
      kind: "Multiply",
      left: { kind: "IntLiteral", intValue: 2n },
      right: { kind: "Input" },
    },
    right: { kind: "IntLiteral", intValue: 1n },
  },
};

export const affineExamples: Example[] = [
  { input: intValue(-2n), output: intValue(-3n) },
  { input: intValue(-1n), output: intValue(-1n) },
  { input: intValue(0n), output: intValue(1n) },
  { input: intValue(1n), output: intValue(3n) },
  { input: intValue(2n), output: intValue(5n) },
];

export function affineConfig(
  overrides: Record<string, unknown> = {},
): ExperimentConfig {
  return parseExperimentConfig(
    JSON.stringify({
      name: "test affine synthesis",
      examples: [
        { input: "-2", output: "-3" },
        { input: "-1", output: "-1" },
        { input: "0", output: "1" },
        { input: "1", output: "3" },
        { input: "2", output: "5" },
      ],
      integerConstants: ["0", "1", "2", "-1", "-2"],
      particles: 8,
      iterations: 6,
      cloneProbability: 0.25,
      essThreshold: 0.6,
      seed: 7,
      lossScale: 2,
      costScale: 0.15,
      lossCap: 1_000,
      maxCost: 12,
      maxDepth: 8,
      maxNodes: 63,
      ...overrides,
    }),
  );
}

export function listConfig(
  name: string,
  output: "List<Int>" | "Int",
  examples: readonly {
    readonly input: readonly string[];
    readonly output: readonly string[] | string;
  }[],
): ExperimentConfig {
  return parseExperimentConfig(
    JSON.stringify({
      name,
      signature: { input: "List<Int>", output },
      examples,
      integerConstants: ["-2", "-1", "0", "1", "2", "3"],
      particles: 8,
      iterations: 7,
      cloneProbability: 0.25,
      essThreshold: 0.6,
      seed: 7,
      lossScale: 2,
      costScale: 0.15,
      lossCap: 1000,
      maxCost: 20,
      maxDepth: 10,
      maxNodes: 127,
    }),
  );
}
