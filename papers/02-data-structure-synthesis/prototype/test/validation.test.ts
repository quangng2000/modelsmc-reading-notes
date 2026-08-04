import assert from "node:assert/strict";
import test from "node:test";

import { INT, STRING, listOf, type Expression } from "../src/ast.js";
import { programMatches } from "../src/synthesis/validation.js";

const INTEGER_SIGNATURE = {
  inputType: listOf(INT),
  outputType: INT,
} as const;

test("full-program validation rejects a lambda with the wrong declared signature", () => {
  const wrongParameterType: Expression = {
    kind: "lambda",
    parameter: "xs",
    parameterType: listOf(STRING),
    body: { kind: "int", value: 7 },
  };

  assert.equal(
    programMatches(
      wrongParameterType,
      { input: [1], output: 7 },
      INTEGER_SIGNATURE,
    ),
    false,
  );
});

test("full-program validation accepts a well-typed matching lambda", () => {
  const matchingProgram: Expression = {
    kind: "lambda",
    parameter: "xs",
    parameterType: listOf(INT),
    body: { kind: "int", value: 7 },
  };

  assert.equal(
    programMatches(
      matchingProgram,
      { input: [1], output: 7 },
      INTEGER_SIGNATURE,
    ),
    true,
  );
});
