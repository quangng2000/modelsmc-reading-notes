import assert from "node:assert/strict";
import test from "node:test";

import {
  BOOL,
  INT,
  STRING,
  functionOf,
  isPrimitiveType,
  listOf,
  primitiveLiteral,
  primitiveTypeOf,
  primitiveValueEquals,
  renderExpression,
  renderPrimitiveValue,
  type Expression,
} from "../src/ast.js";
import { expressionCost } from "../src/cost.js";
import {
  evaluateExpression,
  expectBool,
  expectInt,
  expectString,
} from "../src/evaluation/index.js";
import {
  substituteHole,
  substituteVariable,
} from "../src/synthesis/expressions.js";
import { inferType } from "../src/typecheck.js";

function string(value: string): Expression {
  return { kind: "string", value };
}

function stringList(values: readonly string[]): Expression {
  return {
    kind: "list",
    elementType: STRING,
    elements: values.map(string),
  };
}

test("string primitives have a type, literal form, equality, and escaped rendering", () => {
  assert.equal(isPrimitiveType(STRING), true);
  assert.equal(isPrimitiveType(listOf(STRING)), false);
  assert.deepEqual(primitiveTypeOf("yes"), STRING);
  assert.deepEqual(primitiveTypeOf(true), BOOL);
  assert.deepEqual(primitiveTypeOf(7), INT);
  assert.deepEqual(primitiveLiteral('say "yes"'), {
    kind: "string",
    value: 'say "yes"',
  });
  assert.equal(renderPrimitiveValue('say "yes"'), '"say \\"yes\\""');
  assert.equal(primitiveValueEquals("yes", "yes"), true);
  assert.equal(primitiveValueEquals("yes", "no"), false);
  assert.equal(primitiveValueEquals(1, true), false);
  assert.throws(
    () => primitiveTypeOf(Number.MAX_SAFE_INTEGER + 1),
    /safe integers/,
  );
});

test("concat and length typecheck, evaluate, render, and retain compositional costs", () => {
  const concatenated: Expression = {
    kind: "concat",
    left: string("hello"),
    right: string(" world"),
  };
  const length: Expression = { kind: "length", operand: concatenated };

  assert.deepEqual(inferType(concatenated), STRING);
  assert.equal(expectString(evaluateExpression(concatenated)), "hello world");
  assert.equal(renderExpression(concatenated), '("hello" ++ " world")');
  assert.equal(expressionCost(concatenated), 3);

  assert.deepEqual(inferType(length), INT);
  assert.equal(expectInt(evaluateExpression(length)), 11);
  assert.equal(renderExpression(length), 'length(("hello" ++ " world"))');
  assert.equal(expressionCost(length), 4);
});

test("equality accepts matching primitive types and rejects other combinations", () => {
  const cases: readonly (readonly [Expression, boolean])[] = [
    [
      {
        kind: "comparison",
        operator: "==",
        left: string("same"),
        right: string("same"),
      },
      true,
    ],
    [
      {
        kind: "comparison",
        operator: "==",
        left: { kind: "bool", value: true },
        right: { kind: "bool", value: false },
      },
      false,
    ],
    [
      {
        kind: "comparison",
        operator: "==",
        left: { kind: "int", value: 2 },
        right: { kind: "int", value: 2 },
      },
      true,
    ],
  ];

  for (const [expression, expected] of cases) {
    assert.deepEqual(inferType(expression), BOOL);
    assert.equal(expectBool(evaluateExpression(expression)), expected);
  }

  const mixedEquality: Expression = {
    kind: "comparison",
    operator: "==",
    left: string("1"),
    right: { kind: "int", value: 1 },
  };
  assert.equal(inferType(mixedEquality), undefined);
  assert.throws(() => evaluateExpression(mixedEquality), TypeError);

  assert.equal(
    inferType({
      kind: "comparison",
      operator: "<",
      left: string("a"),
      right: string("b"),
    }),
    undefined,
  );
  assert.equal(
    inferType({
      kind: "comparison",
      operator: "==",
      left: stringList(["a"]),
      right: stringList(["a"]),
    }),
    undefined,
  );
});

test("map supports a list<string> to list<bool> program", () => {
  const mapper: Expression = {
    kind: "lambda",
    parameter: "x",
    parameterType: STRING,
    body: {
      kind: "comparison",
      operator: "==",
      left: { kind: "variable", name: "x" },
      right: string("yes"),
    },
  };
  const program: Expression = {
    kind: "lambda",
    parameter: "xs",
    parameterType: listOf(STRING),
    body: {
      kind: "map",
      mapper,
      list: { kind: "variable", name: "xs" },
    },
  };

  assert.deepEqual(
    inferType(program),
    functionOf(listOf(STRING), listOf(BOOL)),
  );
  assert.equal(
    renderExpression(program),
    '(xs: list<string>) => map((x: string) => (x == "yes"), xs)',
  );
  assert.equal(expressionCost(program), 5);

  if (program.kind !== "lambda") {
    throw new Error("The test program unexpectedly stopped being a lambda.");
  }
  const instantiated = substituteVariable(
    program.body,
    program.parameter,
    stringList(["yes", "no", "yes"]),
  );
  const result = evaluateExpression(instantiated);
  assert.deepEqual(result, [true, false, true]);
  assert.ok(Object.isFrozen(result));
});

test("substitution traverses concat and length nodes", () => {
  const open: Expression = {
    kind: "length",
    operand: {
      kind: "concat",
      left: { kind: "hole", name: "prefix", expectedType: STRING },
      right: { kind: "variable", name: "suffix" },
    },
  };
  const withPrefix = substituteHole(open, "prefix", string("go"));
  const closed = substituteVariable(withPrefix, "suffix", string("pher"));

  assert.equal(expectInt(evaluateExpression(closed)), 6);
});
