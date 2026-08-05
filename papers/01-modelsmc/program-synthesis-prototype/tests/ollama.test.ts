import assert from "node:assert/strict";
import test from "node:test";

import type { Program } from "../src/core/language.verify.js";
import { OllamaProposer } from "../src/shell/ollama/index.js";
import { scoreProgram } from "../src/shell/scoring/index.js";
import { listConfig } from "./support/fixtures.js";

test("the Ollama adapter requests and decodes a complete bounded Program AST", async () => {
  const config = listConfig("map successor", "List<Int>", [
    { input: [], output: [] },
    { input: ["1"], output: ["2"] },
  ]);
  const ancestor: Program = {
    kind: "ExpressionProgram",
    body: { kind: "Input" },
  };
  const ancestorScore = scoreProgram(ancestor, {
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    lossScale: config.lossScale,
    costScale: config.costScale,
    lossCap: config.lossCap,
    maxCost: config.maxCost,
  });
  if (ancestorScore.kind !== "Scored") assert.fail(ancestorScore.reason);

  let requestBody = "";
  const requester: typeof fetch = async (_input, init) => {
    requestBody = String(init?.body ?? "");
    return new Response(
      JSON.stringify({
        choices: [
          {
            finish_reason: "stop",
            message: {
              content: JSON.stringify({
                expression: {
                  kind: "MapProgram",
                  mapper: {
                    kind: "Add",
                    left: { kind: "Item" },
                    right: { kind: "IntLiteral", intValue: "1" },
                  },
                },
                rationale: "increment each scoped item",
              }),
            },
          },
        ],
      }),
      {
        status: 200,
        headers: { "content-type": "application/json" },
      },
    );
  };
  const proposer = new OllamaProposer({ requester, timeoutMs: 1000 });
  const proposal = await proposer.propose({
    requestIndex: 0,
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    integerConstants: config.integerConstants,
    maxDepth: config.maxDepth,
    maxNodes: config.maxNodes,
    maxCost: config.maxCost,
    ancestor,
    ancestorScore,
    ancestorFeedback: "proposal rejected: expression cost 33 exceeds maximum 30",
  });

  assert.equal(proposal.expression.kind, "MapProgram");
  assert.equal(proposal.rationale, "increment each scoped item");
  assert.match(requestBody, /typed_program_proposal/);
  assert.match(requestBody, /FoldRightProgram/);
  assert.match(requestBody, /must not reference the outer Input/);
  assert.match(requestBody, /Previous transition feedback/);
  assert.match(requestBody, /replace enumerated literal cases with compact relational predicates/);
  assert.equal(
    Object.hasOwn(JSON.parse(requestBody) as object, "reasoning_effort"),
    false,
    "reasoning_effort must be omitted unless the caller enables it",
  );
});

test("the Ollama adapter sends reasoning effort only when explicitly enabled", async () => {
  let requestBody = "";
  const requester: typeof fetch = async (_input, init) => {
    requestBody = String(init?.body ?? "");
    return new Response(
      JSON.stringify({
        choices: [
          {
            finish_reason: "stop",
            message: {
              content: JSON.stringify({
                expression: { kind: "ExpressionProgram", body: { kind: "Input" } },
                rationale: "identity",
              }),
            },
          },
        ],
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  };
  const config = listConfig("identity", "List<Int>", [{ input: [], output: [] }]);
  const ancestor: Program = {
    kind: "ExpressionProgram",
    body: { kind: "Input" },
  };
  const ancestorScore = scoreProgram(ancestor, {
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    lossScale: config.lossScale,
    costScale: config.costScale,
    lossCap: config.lossCap,
    maxCost: config.maxCost,
  });
  if (ancestorScore.kind !== "Scored") assert.fail(ancestorScore.reason);

  await new OllamaProposer({
    requester,
    timeoutMs: 1000,
    reasoningEffort: "low",
  }).propose({
    requestIndex: 0,
    inputType: config.inputType,
    outputType: config.outputType,
    examples: config.examples,
    integerConstants: config.integerConstants,
    maxDepth: config.maxDepth,
    maxNodes: config.maxNodes,
    maxCost: config.maxCost,
    ancestor,
    ancestorScore,
    ancestorFeedback: "initial seed",
  });

  assert.equal(
    (JSON.parse(requestBody) as { reasoning_effort?: unknown }).reasoning_effort,
    "low",
  );
});
