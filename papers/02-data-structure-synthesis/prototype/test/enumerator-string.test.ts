import assert from "node:assert/strict";
import test from "node:test";

import {
  BOOL,
  INT,
  STRING,
  renderExpression,
} from "../src/ast.js";
import { expressionCost } from "../src/cost.js";
import {
  enumerateExpressionsByCost,
  evaluateWithTyped,
} from "../src/enumeration/index.js";
import { inferType } from "../src/typecheck.js";

test("enumerates typed string variables, literals, concat, and length by exact cost", () => {
  const stringBuckets = [
    ...enumerateExpressionsByCost({
      maxCost: 1,
      variables: [{ name: "s", type: "string" }],
      targetType: "string",
    }),
  ];

  assert.deepEqual(
    stringBuckets[0]?.expressions.map(renderExpression),
    ["s"],
  );
  assert.deepEqual(
    stringBuckets[1]?.expressions.map(renderExpression),
    ['""', "(s ++ s)"],
  );

  const intBuckets = [
    ...enumerateExpressionsByCost({
      maxCost: 1,
      variables: [{ name: "s", type: "string" }],
      targetType: "int",
    }),
  ];
  const renderedInts = intBuckets[1]?.expressions.map(renderExpression);
  assert.ok(renderedInts?.includes("length(s)"));

  for (const bucket of stringBuckets) {
    for (const expression of bucket.expressions) {
      assert.equal(expressionCost(expression), bucket.cost);
      assert.deepEqual(
        inferType(expression, [{ name: "s", type: STRING }]),
        STRING,
      );
    }
  }
  for (const bucket of intBuckets) {
    for (const expression of bucket.expressions) {
      assert.equal(expressionCost(expression), bucket.cost);
      assert.deepEqual(
        inferType(expression, [{ name: "s", type: STRING }]),
        INT,
      );
    }
  }
});

test("enumerates same-type equality for typed bool and string variables", () => {
  const boolExpressions = [
    ...enumerateExpressionsByCost({
      maxCost: 1,
      variables: [{ name: "b", type: "bool" }],
      targetType: "bool",
    }),
  ][1]?.expressions;
  assert.ok(boolExpressions);
  const boolEquality = boolExpressions.find(
    (expression) =>
      expression.kind === "comparison" &&
      expression.operator === "==" &&
      expression.left.kind === "variable" &&
      expression.right.kind === "variable",
  );
  assert.ok(boolEquality);
  assert.deepEqual(inferType(boolEquality, [{ name: "b", type: BOOL }]), BOOL);

  const stringExpressions = [
    ...enumerateExpressionsByCost({
      maxCost: 1,
      variables: [{ name: "s", type: "string" }],
      targetType: "bool",
    }),
  ][1]?.expressions;
  assert.ok(stringExpressions);
  const stringEquality = stringExpressions.find(
    (expression) =>
      expression.kind === "comparison" &&
      expression.operator === "==" &&
      expression.left.kind === "variable" &&
      expression.right.kind === "variable",
  );
  assert.ok(stringEquality);
  assert.deepEqual(
    inferType(stringEquality, [{ name: "s", type: STRING }]),
    BOOL,
  );
});

test("composes generated string predicates with bool equality at cost 5", () => {
  const buckets = [
    ...enumerateExpressionsByCost({
      maxCost: 5,
      variables: [{ name: "x", type: "string" }],
      targetType: "bool",
      stringConstants: ["a", "b"],
    }),
  ];

  assert.ok(
    buckets[5]?.expressions
      .map(renderExpression)
      .includes('((x == "a") == (x == "b"))'),
  );
});

test("uses the default empty string only for string-aware searches", () => {
  const legacy = [
    ...enumerateExpressionsByCost({ maxCost: 1, targetType: "int" }),
  ];
  assert.deepEqual(
    legacy[1]?.expressions.map(renderExpression),
    ["-1", "0", "1", "2", "(x + x)", "(x - x)"],
  );
  assert.ok(
    legacy.every(({ expressions }) =>
      expressions.every(
        (expression) =>
          expression.kind !== "string" &&
          expression.kind !== "concat" &&
          expression.kind !== "length",
      ),
    ),
  );

  const withDefault = [
    ...enumerateExpressionsByCost({
      maxCost: 1,
      variables: [{ name: "x", type: "int" }],
      targetType: "string",
    }),
  ];
  assert.deepEqual(withDefault[1]?.expressions.map(renderExpression), ['""']);

  const overridden = [
    ...enumerateExpressionsByCost({
      maxCost: 1,
      variables: [{ name: "s", type: "string" }],
      targetType: "string",
      stringConstants: ["seed"],
    }),
  ];
  assert.deepEqual(
    overridden[1]?.expressions.map(renderExpression),
    ['"seed"', "(s ++ s)"],
  );
});

test("evaluates string candidates with primitive-name and object-type bindings", () => {
  const concat = {
    kind: "concat",
    left: { kind: "variable", name: "s" },
    right: { kind: "string", value: "!" },
  } as const;
  assert.equal(
    evaluateWithTyped(concat, [{ name: "s", type: "string", value: "hi" }]),
    "hi!",
  );

  const length = {
    kind: "length",
    operand: { kind: "variable", name: "s" },
  } as const;
  assert.equal(
    evaluateWithTyped(length, [{ name: "s", type: STRING, value: "hello" }]),
    5,
  );
});

test("rejects duplicate mixed-form variables and non-string string constants", () => {
  assert.throws(
    () => [
      ...enumerateExpressionsByCost({
        maxCost: 0,
        variables: ["x", { name: "x", type: "string" }],
      }),
    ],
    /variables must be distinct/,
  );
  assert.throws(
    () => [
      ...enumerateExpressionsByCost({
        maxCost: 0,
        stringConstants: [1] as unknown as readonly string[],
      }),
    ],
    /stringConstants must contain only strings/,
  );
});
