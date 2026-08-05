/**
 * Validate that an intended target program exactly solves a task spec and fits
 * its cost budget — run this before spending proposal budget on a new task.
 *
 * Usage: npx tsx experiments/validate-task.ts
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { Program } from "../src/core/language.verify.js";
import { parseExperimentConfig } from "../src/shell/config/index.js";
import { scoreProgram } from "../src/shell/scoring/index.js";
import { renderProgram } from "../src/shell/ast/render.js";

const ROOT = resolve(import.meta.dirname, "..");

const item = { kind: "Item" } as const;
const acc = { kind: "Accumulator" } as const;
function lit(value: bigint) {
  return { kind: "IntLiteral", intValue: value } as const;
}

interface TargetCase {
  readonly spec: string;
  readonly program: Program;
}

const CASES: TargetCase[] = [
  {
    spec: "examples/foldr-signed-window.json",
    // if x < 0 then (if -3 < x then (0 - x) :: acc else acc)
    //          else (if x < 3 then (x * x) :: acc else acc)
    program: {
      kind: "FoldRightProgram",
      initial: { kind: "EmptyIntList" },
      reducer: {
        kind: "IfThenElse",
        condition: { kind: "LessThan", left: item, right: lit(0n) },
        thenExpr: {
          kind: "IfThenElse",
          condition: { kind: "LessThan", left: lit(-3n), right: item },
          thenExpr: {
            kind: "PrependInt",
            head: { kind: "Subtract", left: lit(0n), right: item },
            tail: acc,
          },
          elseExpr: acc,
        },
        elseExpr: {
          kind: "IfThenElse",
          condition: { kind: "LessThan", left: item, right: lit(3n) },
          thenExpr: {
            kind: "PrependInt",
            head: { kind: "Multiply", left: item, right: item },
            tail: acc,
          },
          elseExpr: acc,
        },
      },
    },
  },
  {
    spec: "examples/foldr-window-penalty-sum.json",
    // if -2 < x then (if x < 3 then (x*x) + acc else acc - x) else acc
    program: {
      kind: "FoldRightProgram",
      initial: lit(0n),
      reducer: {
        kind: "IfThenElse",
        condition: { kind: "LessThan", left: lit(-2n), right: item },
        thenExpr: {
          kind: "IfThenElse",
          condition: { kind: "LessThan", left: item, right: lit(3n) },
          thenExpr: {
            kind: "Add",
            left: { kind: "Multiply", left: item, right: item },
            right: acc,
          },
          elseExpr: { kind: "Subtract", left: acc, right: item },
        },
        elseExpr: acc,
      },
    },
  },
];

let failed = false;
for (const testCase of CASES) {
  const config = parseExperimentConfig(readFileSync(resolve(ROOT, testCase.spec), "utf8"));
  const score = scoreProgram(testCase.program, {
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    lossScale: config.lossScale,
    costScale: config.costScale,
    lossCap: config.lossCap,
    maxCost: config.maxCost,
  });
  console.log(`\n${testCase.spec}`);
  console.log(`  target: ${renderProgram(testCase.program, config.inputType)}`);
  if (score.kind !== "Scored") {
    console.log(`  REJECTED: ${score.reason}`);
    failed = true;
    continue;
  }
  console.log(
    `  cost=${score.cost}/${config.maxCost} loss=${score.totalLoss} exact=${score.exactMatches}/${config.examples.length} exactProgram=${score.exactProgram}`,
  );
  if (!score.exactProgram) {
    failed = true;
    for (const evaluation of score.evaluations.filter((entry) => !entry.exact)) {
      console.log(`  MISMATCH: input=${JSON.stringify(evaluation.input)} loss=${evaluation.loss}`);
    }
  }
}
process.exitCode = failed ? 1 : 0;
