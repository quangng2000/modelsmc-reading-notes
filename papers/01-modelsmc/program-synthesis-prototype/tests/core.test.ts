import assert from "node:assert/strict";
import test from "node:test";

import {
  acceptProgram,
  boolCons,
  boolListValue,
  boolNil,
  boolValue,
  evaluate,
  examplesHaveSignature,
  expressionBodyCost,
  expressionCost,
  inferType,
  intCons,
  intListValue,
  intNil,
  intValue,
  matchesAllExamples,
  matchesExample,
  sameValue,
  type BoolList,
  type Example,
  type Expr,
  type IntList,
  type Program,
  type RuntimeValue,
} from "../src/core/language.verify.js";

function expression(body: Expr): Program {
  return { kind: "ExpressionProgram", body };
}

function intList(items: readonly bigint[]): IntList {
  let result = intNil();
  for (let index = items.length - 1; index >= 0; index -= 1) {
    result = intCons(items[index]!, result);
  }
  return result;
}

function boolList(items: readonly boolean[]): BoolList {
  let result = boolNil();
  for (let index = items.length - 1; index >= 0; index -= 1) {
    result = boolCons(items[index]!, result);
  }
  return result;
}

function ints(...items: bigint[]): RuntimeValue {
  return intListValue(intList(items));
}

function bools(...items: boolean[]): RuntimeValue {
  return boolListValue(boolList(items));
}

const input: Expr = { kind: "Input" };

const twicePlusOneBody: Expr = {
  kind: "Add",
  left: {
    kind: "Multiply",
    left: { kind: "IntLiteral", intValue: 2n },
    right: input,
  },
  right: { kind: "IntLiteral", intValue: 1n },
};
const twicePlusOne = expression(twicePlusOneBody);

const linearExamples: Example[] = [
  { input: intValue(-2n), output: intValue(-3n) },
  { input: intValue(-1n), output: intValue(-1n) },
  { input: intValue(0n), output: intValue(1n) },
  { input: intValue(1n), output: intValue(3n) },
  { input: intValue(4n), output: intValue(9n) },
];

test("infers and evaluates wrapped scalar programs", () => {
  assert.deepEqual(inferType(twicePlusOne, "IntType"), {
    kind: "TypeOk",
    inferred: "IntType",
  });
  assert.deepEqual(evaluate(twicePlusOne, intValue(5n)), {
    kind: "EvalOk",
    output: intValue(11n),
  });

  const booleanProgram = expression({
    kind: "And",
    left: { kind: "Not", operand: input },
    right: { kind: "BoolLiteral", boolValue: true },
  });
  assert.deepEqual(inferType(booleanProgram, "BoolType"), {
    kind: "TypeOk",
    inferred: "BoolType",
  });
  assert.deepEqual(evaluate(booleanProgram, boolValue(false)), {
    kind: "EvalOk",
    output: boolValue(true),
  });
});

test("rejects ill-typed and ill-scoped expression programs", () => {
  const illTyped = expression({
    kind: "Add",
    left: { kind: "BoolLiteral", boolValue: true },
    right: input,
  });
  const unboundItem = expression({ kind: "Item" });
  const unboundAccumulator = expression({ kind: "Accumulator" });

  assert.deepEqual(inferType(illTyped, "IntType"), { kind: "TypeError" });
  assert.deepEqual(evaluate(illTyped, intValue(2n)), { kind: "EvalError" });
  assert.deepEqual(inferType(unboundItem, "IntType"), { kind: "TypeError" });
  assert.deepEqual(evaluate(unboundItem, intValue(2n)), { kind: "EvalError" });
  assert.deepEqual(inferType(unboundAccumulator, "IntType"), { kind: "TypeError" });
  assert.deepEqual(evaluate(unboundAccumulator, intValue(2n)), { kind: "EvalError" });
});

test("accepts an exact scalar program and rejects a near miss", () => {
  const nearMiss = { input: intValue(2n), output: intValue(6n) };
  assert.equal(examplesHaveSignature(linearExamples), true);
  assert.equal(matchesAllExamples(twicePlusOne, linearExamples), true);
  assert.equal(acceptProgram(twicePlusOne, linearExamples), true);
  assert.equal(matchesExample(twicePlusOne, nearMiss), false);
  assert.equal(acceptProgram(twicePlusOne, [...linearExamples, nearMiss]), false);
});

test("map preserves recursive-list shape and can change element type", () => {
  const increment: Program = {
    kind: "MapProgram",
    mapper: {
      kind: "Add",
      left: { kind: "Item" },
      right: { kind: "IntLiteral", intValue: 1n },
    },
  };
  const negative: Program = {
    kind: "MapProgram",
    mapper: {
      kind: "LessThan",
      left: { kind: "Item" },
      right: { kind: "IntLiteral", intValue: 0n },
    },
  };

  assert.deepEqual(inferType(increment, "IntListType"), {
    kind: "TypeOk",
    inferred: "IntListType",
  });
  assert.deepEqual(evaluate(increment, ints()), {
    kind: "EvalOk",
    output: ints(),
  });
  assert.deepEqual(evaluate(increment, ints(-2n, 0n, 3n)), {
    kind: "EvalOk",
    output: ints(-1n, 1n, 4n),
  });
  assert.deepEqual(inferType(negative, "IntListType"), {
    kind: "TypeOk",
    inferred: "BoolListType",
  });
  assert.deepEqual(evaluate(negative, ints(-2n, 0n, 3n)), {
    kind: "EvalOk",
    output: bools(true, false, false),
  });
});

test("map bodies are scoped and cannot inspect the whole outer list", () => {
  const wholeInputMapper: Program = {
    kind: "MapProgram",
    mapper: { kind: "Input" },
  };
  const accumulatorMapper: Program = {
    kind: "MapProgram",
    mapper: { kind: "Accumulator" },
  };

  assert.deepEqual(inferType(wholeInputMapper, "IntListType"), {
    kind: "TypeError",
  });
  assert.deepEqual(inferType(accumulatorMapper, "IntListType"), {
    kind: "TypeError",
  });
});

test("foldr sums integer lists and follows right-associative order", () => {
  const sum: Program = {
    kind: "FoldRightProgram",
    initial: { kind: "IntLiteral", intValue: 0n },
    reducer: {
      kind: "Add",
      left: { kind: "Item" },
      right: { kind: "Accumulator" },
    },
  };
  const subtractRight: Program = {
    kind: "FoldRightProgram",
    initial: { kind: "IntLiteral", intValue: 0n },
    reducer: {
      kind: "Subtract",
      left: { kind: "Item" },
      right: { kind: "Accumulator" },
    },
  };

  assert.deepEqual(inferType(sum, "IntListType"), {
    kind: "TypeOk",
    inferred: "IntType",
  });
  assert.deepEqual(evaluate(sum, ints(1n, 2n, 3n)), {
    kind: "EvalOk",
    output: intValue(6n),
  });
  assert.deepEqual(evaluate(subtractRight, ints(1n, 2n, 3n)), {
    kind: "EvalOk",
    output: intValue(2n),
  });
});

test("foldr can construct a recursive list with typed Nil and Cons", () => {
  const positives: Program = {
    kind: "FoldRightProgram",
    initial: { kind: "EmptyIntList" },
    reducer: {
      kind: "IfThenElse",
      condition: {
        kind: "LessThan",
        left: { kind: "IntLiteral", intValue: 0n },
        right: { kind: "Item" },
      },
      thenExpr: {
        kind: "PrependInt",
        head: { kind: "Item" },
        tail: { kind: "Accumulator" },
      },
      elseExpr: { kind: "Accumulator" },
    },
  };

  assert.deepEqual(inferType(positives, "IntListType"), {
    kind: "TypeOk",
    inferred: "IntListType",
  });
  assert.deepEqual(evaluate(positives, ints(-2n, 0n, 3n, 4n)), {
    kind: "EvalOk",
    output: ints(3n, 4n),
  });
});

test("foldr rejects mismatched accumulators and outer-input-dependent bodies", () => {
  const mismatch: Program = {
    kind: "FoldRightProgram",
    initial: { kind: "IntLiteral", intValue: 0n },
    reducer: { kind: "BoolLiteral", boolValue: true },
  };
  const outerDependent: Program = {
    kind: "FoldRightProgram",
    initial: { kind: "IntLiteral", intValue: 0n },
    reducer: { kind: "Input" },
  };
  const badCons = expression({
    kind: "PrependInt",
    head: { kind: "BoolLiteral", boolValue: true },
    tail: { kind: "EmptyIntList" },
  });

  assert.deepEqual(inferType(mismatch, "IntListType"), { kind: "TypeError" });
  assert.deepEqual(inferType(outerDependent, "IntListType"), { kind: "TypeError" });
  assert.deepEqual(inferType(badCons, "IntType"), { kind: "TypeError" });
});

test("list equality and signatures include element-list tags", () => {
  assert.equal(sameValue(ints(1n, 2n), ints(1n, 2n)), true);
  assert.equal(sameValue(ints(1n, 2n), ints(1n, 3n)), false);
  assert.equal(sameValue(ints(), bools()), false);
  assert.equal(
    examplesHaveSignature([
      { input: ints(), output: ints() },
      { input: ints(1n), output: ints(2n) },
    ]),
    true,
  );
  assert.equal(
    examplesHaveSignature([
      { input: ints(), output: ints() },
      { input: bools(), output: ints() },
    ]),
    false,
  );
});

test("exact PBE acceptance works for map examples", () => {
  const increment: Program = {
    kind: "MapProgram",
    mapper: {
      kind: "Add",
      left: { kind: "Item" },
      right: { kind: "IntLiteral", intValue: 1n },
    },
  };
  const examples: Example[] = [
    { input: ints(), output: ints() },
    { input: ints(1n), output: ints(2n) },
    { input: ints(-2n, 0n, 3n), output: ints(-1n, 1n, 4n) },
  ];

  assert.equal(acceptProgram(increment, examples), true);
  assert.equal(
    acceptProgram(increment, [
      ...examples,
      { input: ints(4n), output: ints(4n) },
    ]),
    false,
  );
});

test("structural costs cover expression bodies and combinator wrappers", () => {
  const map: Program = {
    kind: "MapProgram",
    mapper: {
      kind: "Add",
      left: { kind: "Item" },
      right: { kind: "IntLiteral", intValue: 1n },
    },
  };
  const fold: Program = {
    kind: "FoldRightProgram",
    initial: { kind: "IntLiteral", intValue: 0n },
    reducer: {
      kind: "Add",
      left: { kind: "Item" },
      right: { kind: "Accumulator" },
    },
  };

  assert.equal(expressionBodyCost(input), 1n);
  assert.equal(expressionBodyCost(twicePlusOneBody), 5n);
  assert.equal(expressionCost(twicePlusOne), 5n);
  assert.equal(expressionCost(map), 5n);
  assert.equal(expressionCost(fold), 7n);
  assert.ok(expressionCost(map) > expressionBodyCost(map.mapper));
  assert.ok(expressionCost(fold) > expressionBodyCost(fold.reducer));
});

test("recursive integer lists preserve bigint values beyond safe integers", () => {
  const beyondSafeInteger = 9_007_199_254_740_993n;
  const increment: Program = {
    kind: "MapProgram",
    mapper: {
      kind: "Add",
      left: { kind: "Item" },
      right: { kind: "IntLiteral", intValue: 1n },
    },
  };

  assert.deepEqual(evaluate(increment, ints(beyondSafeInteger)), {
    kind: "EvalOk",
    output: ints(9_007_199_254_740_994n),
  });
});
