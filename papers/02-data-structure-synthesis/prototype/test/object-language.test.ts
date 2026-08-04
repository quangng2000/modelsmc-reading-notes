import assert from "node:assert/strict";
import test from "node:test";

import {
  BOOL,
  INT,
  functionOf,
  type ComparisonOperator,
  type Expression,
} from "../src/ast.js";
import {
  evaluateExpression,
  expectBool,
  expectInt,
  expectIntList,
} from "../src/evaluation/index.js";
import { inferType } from "../src/typecheck.js";

function int(value: number): Expression {
  return { kind: "int", value };
}

function bool(value: boolean): Expression {
  return { kind: "bool", value };
}

function compare(
  operator: ComparisonOperator,
  left: Expression,
  right: Expression,
): Expression {
  return { kind: "comparison", operator, left, right };
}

function intList(values: readonly number[]): Expression {
  return { kind: "list", elementType: INT, elements: values.map(int) };
}

// (x: int) => (0 < x)
const positivePredicate: Expression = {
  kind: "lambda",
  parameter: "x",
  parameterType: INT,
  body: compare("<", int(0), { kind: "variable", name: "x" }),
};

test("bool literals typecheck to bool and evaluate to themselves", () => {
  assert.deepEqual(inferType(bool(true)), BOOL);
  assert.deepEqual(inferType(bool(false)), BOOL);
  assert.equal(evaluateExpression(bool(true)), true);
  assert.equal(evaluateExpression(bool(false)), false);
});

test("each comparison operator typechecks to bool and evaluates over ints", () => {
  const cases: readonly (readonly [
    ComparisonOperator,
    number,
    number,
    boolean,
  ])[] = [
    ["<", 1, 2, true],
    ["<", 2, 2, false],
    ["<=", 2, 2, true],
    ["<=", 3, 2, false],
    ["==", 2, 2, true],
    ["==", 1, 2, false],
  ];

  for (const [operator, left, right, expected] of cases) {
    const expression = compare(operator, int(left), int(right));
    assert.deepEqual(inferType(expression), BOOL);
    assert.equal(expectBool(evaluateExpression(expression)), expected);
  }
});

test("logic evaluates both operands and reports both truth tables", () => {
  assert.equal(
    evaluateExpression({
      kind: "logic",
      operator: "&&",
      left: bool(true),
      right: bool(false),
    }),
    false,
  );
  assert.equal(
    evaluateExpression({
      kind: "logic",
      operator: "&&",
      left: bool(true),
      right: bool(true),
    }),
    true,
  );
  assert.equal(
    evaluateExpression({
      kind: "logic",
      operator: "||",
      left: bool(false),
      right: bool(true),
    }),
    true,
  );
  assert.equal(
    evaluateExpression({
      kind: "logic",
      operator: "||",
      left: bool(false),
      right: bool(false),
    }),
    false,
  );

  // STRICT evaluation: no short-circuit exists to hide the RangeError in the
  // left operand, even though the right operand is already false.
  const moduloByZero = compare(
    "==",
    { kind: "binary", operator: "%", left: int(1), right: int(0) },
    int(0),
  );
  assert.throws(
    () =>
      evaluateExpression({
        kind: "logic",
        operator: "&&",
        left: moduloByZero,
        right: bool(false),
      }),
    RangeError,
  );
});

test("not negates its boolean operand", () => {
  assert.deepEqual(inferType({ kind: "not", operand: bool(false) }), BOOL);
  assert.equal(evaluateExpression({ kind: "not", operand: bool(true) }), false);
  assert.equal(
    evaluateExpression({ kind: "not", operand: bool(false) }),
    true,
  );
});

test("modulo follows the dividend's sign and rejects a zero divisor", () => {
  const modulo = (left: number, right: number): number =>
    expectInt(
      evaluateExpression({
        kind: "binary",
        operator: "%",
        left: int(left),
        right: int(right),
      }),
    );

  assert.equal(modulo(7, 3), 1);
  assert.equal(modulo(-7, 3), -1);
  assert.equal(modulo(7, -3), 1);
  assert.throws(() => modulo(1, 0), RangeError);
  assert.throws(() => modulo(1, 0), /Modulo by zero/);
});

test("filter preserves order, freezes its result, and needs a bool predicate", () => {
  const filtered = evaluateExpression({
    kind: "filter",
    predicate: positivePredicate,
    list: intList([1, -2, 3, 0]),
  });

  assert.deepEqual(expectIntList(filtered), [1, 3]);
  assert.ok(Object.isFrozen(filtered));

  // (x: int) => x returns int, not bool, so the whole filter is ill-typed.
  const intPredicate: Expression = {
    kind: "lambda",
    parameter: "x",
    parameterType: INT,
    body: { kind: "variable", name: "x" },
  };
  assert.throws(
    () =>
      evaluateExpression({
        kind: "filter",
        predicate: intPredicate,
        list: intList([1]),
      }),
    TypeError,
  );
});

test("foldl folds from the left through the curried reducer", () => {
  // (acc: int) => (x: int) => (acc - x)
  const reducer: Expression = {
    kind: "lambda",
    parameter: "acc",
    parameterType: INT,
    body: {
      kind: "lambda",
      parameter: "x",
      parameterType: INT,
      body: {
        kind: "binary",
        operator: "-",
        left: { kind: "variable", name: "acc" },
        right: { kind: "variable", name: "x" },
      },
    },
  };

  // Curried application: the reducer typechecks as int -> (int -> int).
  assert.deepEqual(inferType(reducer), functionOf(INT, functionOf(INT, INT)));

  // Left associativity: ((0 - 1) - 2) - 3 = -6. A right fold would instead
  // give 1 - (2 - 3) = 2.
  const folded = evaluateExpression({
    kind: "fold",
    reducer,
    initial: int(0),
    list: intList([1, 2, 3]),
  });
  assert.equal(expectInt(folded), -6);
});

test("rejects ill-typed uses of the new constructs", () => {
  assert.equal(
    inferType({ kind: "logic", operator: "&&", left: int(1), right: int(2) }),
    undefined,
  );
  assert.equal(
    inferType({
      kind: "binary",
      operator: "+",
      left: bool(true),
      right: int(1),
    }),
    undefined,
  );
  assert.equal(
    inferType({
      kind: "filter",
      predicate: positivePredicate,
      list: int(3),
    }),
    undefined,
  );

  // A non-curried reducer (int -> int) cannot type a fold.
  const flatReducer: Expression = {
    kind: "lambda",
    parameter: "acc",
    parameterType: INT,
    body: { kind: "variable", name: "acc" },
  };
  assert.equal(
    inferType({
      kind: "fold",
      reducer: flatReducer,
      initial: int(0),
      list: intList([1]),
    }),
    undefined,
  );
});
