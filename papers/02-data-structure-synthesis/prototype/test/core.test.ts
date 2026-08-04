import assert from "node:assert/strict";
import test from "node:test";

import {
  INT,
  functionOf,
  listOf,
  renderExpression,
  typeEquals,
  type Expression,
} from "../src/ast.js";
import { expressionCost } from "../src/cost.js";
import {
  evaluateExpression,
  expectIntList,
  type Value,
  type ValueBinding,
} from "../src/evaluation/index.js";
import {
  MAP_SKELETON,
  substituteHole,
} from "../src/synthesis/index.js";
import { inferType } from "../src/typecheck.js";

test("renders and type-checks the open map skeleton", () => {
  assert.equal(
    renderExpression(MAP_SKELETON),
    "(xs: list<int>) => map(?f: int -> int, xs)",
  );

  const actualType = inferType(MAP_SKELETON);
  const expectedType = functionOf(listOf(INT), listOf(INT));
  assert.ok(actualType);
  assert.ok(typeEquals(actualType, expectedType));
});

test("rejects an object-language type error", () => {
  const malformed: Expression = {
    kind: "binary",
    operator: "+",
    left: { kind: "int", value: 1 },
    right: {
      kind: "list",
      elementType: INT,
      elements: [{ kind: "int", value: 2 }],
    },
  };

  assert.equal(inferType(malformed), undefined);
  assert.throws(() => evaluateExpression(malformed), TypeError);
});

test("computes skeleton lower bounds and completed-program costs", () => {
  const mapper: Expression = {
    kind: "lambda",
    parameter: "x",
    parameterType: INT,
    body: {
      kind: "binary",
      operator: "+",
      left: { kind: "variable", name: "x" },
      right: { kind: "int", value: 2 },
    },
  };
  const completed = substituteHole(MAP_SKELETON, "f", mapper);

  assert.equal(expressionCost(MAP_SKELETON), 2);
  assert.equal(expressionCost(completed), 5);
  assert.ok(expressionCost(MAP_SKELETON) <= expressionCost(completed));
});

test("evaluates a typed map expression", () => {
  const expression: Expression = {
    kind: "map",
    mapper: {
      kind: "lambda",
      parameter: "x",
      parameterType: INT,
      body: {
        kind: "binary",
        operator: "+",
        left: { kind: "variable", name: "x" },
        right: { kind: "int", value: 2 },
      },
    },
    list: {
      kind: "list",
      elementType: INT,
      elements: [
        { kind: "int", value: 1 },
        { kind: "int", value: 2 },
      ],
    },
  };

  assert.deepEqual(expectIntList(evaluateExpression(expression)), [3, 4]);
  assert.throws(
    () =>
      evaluateExpression({
        kind: "hole",
        name: "f",
        expectedType: functionOf(INT, INT),
      }),
    /unresolved hole/,
  );
});

test("rejects holes hidden beneath lambdas or empty maps", () => {
  const openMapper: Expression = {
    kind: "lambda",
    parameter: "x",
    parameterType: INT,
    body: { kind: "hole", name: "h", expectedType: INT },
  };

  assert.throws(() => evaluateExpression(openMapper), /unresolved holes/);
  assert.throws(
    () =>
      evaluateExpression({
        kind: "map",
        mapper: openMapper,
        list: { kind: "list", elementType: INT, elements: [] },
      }),
    /unresolved holes/,
  );
});

test("snapshots captured environments for lexical closures", () => {
  const captured: ValueBinding[] = [{ name: "y", type: INT, value: 1 }];
  const closure = evaluateExpression(
    {
      kind: "lambda",
      parameter: "x",
      parameterType: INT,
      body: { kind: "variable", name: "y" },
    },
    captured,
  );
  captured.length = 0;

  const appliedByMap: Expression = {
    kind: "map",
    mapper: { kind: "variable", name: "f" },
    list: {
      kind: "list",
      elementType: INT,
      elements: [{ kind: "int", value: 7 }],
    },
  };

  assert.deepEqual(
    expectIntList(
      evaluateExpression(appliedByMap, [
        {
          name: "f",
          type: functionOf(INT, INT),
          value: closure,
        },
      ]),
    ),
    [1],
  );
});

test("snapshots a closure body and rejects branded-looking copies", () => {
  const retainedBody = { kind: "int" as const, value: 7 };
  const closure = evaluateExpression({
    kind: "lambda",
    parameter: "x",
    parameterType: INT,
    body: retainedBody,
  });
  retainedBody.value = 1.5;

  const mapped: Expression = {
    kind: "map",
    mapper: { kind: "variable", name: "f" },
    list: {
      kind: "list",
      elementType: INT,
      elements: [{ kind: "int", value: 1 }],
    },
  };
  assert.deepEqual(
    expectIntList(
      evaluateExpression(mapped, [
        { name: "f", type: functionOf(INT, INT), value: closure },
      ]),
    ),
    [7],
  );

  const copiedClosure = {
    ...(closure as object),
    body: { kind: "variable", name: "missing" },
  } as unknown as Value;
  assert.throws(
    () =>
      evaluateExpression(mapped, [
        {
          name: "f",
          type: functionOf(INT, INT),
          value: copiedClosure,
        },
      ]),
    /not created by the evaluator/,
  );
});

test("rejects function values not created by the evaluator", () => {
  const forged = {
    kind: "closure",
    parameter: "x",
    parameterType: INT,
    resultType: INT,
    body: { kind: "variable", name: "missing" },
    environment: [],
  } as unknown as Value;
  const mapped: Expression = {
    kind: "map",
    mapper: { kind: "variable", name: "f" },
    list: {
      kind: "list",
      elementType: INT,
      elements: [{ kind: "int", value: 1 }],
    },
  };

  assert.throws(
    () =>
      evaluateExpression(mapped, [
        { name: "f", type: functionOf(INT, INT), value: forged },
      ]),
    /not created by the evaluator/,
  );
  assert.throws(
    () =>
      evaluateExpression(mapped, [
        {
          name: "f",
          type: functionOf(INT, INT),
          value: null as unknown as Value,
        },
      ]),
    /not created by the evaluator/,
  );
});

test("fills a hole using variables from its lexical type context", () => {
  const open: Expression = {
    kind: "lambda",
    parameter: "x",
    parameterType: INT,
    body: { kind: "hole", name: "h", expectedType: INT },
  };
  const completed = substituteHole(open, "h", {
    kind: "variable",
    name: "x",
  });

  assert.equal(renderExpression(completed), "(x: int) => x");
  const completedType = inferType(completed);
  assert.ok(completedType);
  assert.ok(typeEquals(completedType, functionOf(INT, INT)));
});
