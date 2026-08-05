import assert from "node:assert/strict";
import test from "node:test";

import {
  CliArgumentError,
  parseCliArgs,
} from "../src/shell/cli/index.js";

test("reasoning effort is omitted by default and parsed only when requested", () => {
  const defaults = parseCliArgs(["spec.json"]);
  if (defaults === "help") assert.fail("expected parsed CLI options");
  assert.equal(Object.hasOwn(defaults, "reasoningEffort"), false);

  for (const effort of ["low", "medium", "high"] as const) {
    const options = parseCliArgs([
      "spec.json",
      "--reasoning-effort",
      effort,
    ]);
    if (options === "help") assert.fail("expected parsed CLI options");
    assert.equal(options.reasoningEffort, effort);
  }

  assert.throws(
    () => parseCliArgs(["spec.json", "--reasoning-effort", "none"]),
    (error: unknown) =>
      error instanceof CliArgumentError &&
      /must be low, medium, or high/.test(error.message),
  );
});

test("the model default depends on the selected proposal backend", () => {
  const ollama = parseCliArgs(["spec.json", "--proposal", "ollama"]);
  if (ollama === "help") assert.fail("expected parsed CLI options");
  assert.equal(ollama.model, "gpt-oss:20b");

  const anthropic = parseCliArgs(["spec.json", "--proposal", "anthropic"]);
  if (anthropic === "help") assert.fail("expected parsed CLI options");
  assert.equal(anthropic.model, "claude-sonnet-5");
  assert.equal(anthropic.anthropicUrl, "https://api.anthropic.com");

  const explicit = parseCliArgs([
    "spec.json",
    "--proposal",
    "anthropic",
    "--model",
    "claude-opus-5",
    "--anthropic-url",
    "https://example.test",
  ]);
  if (explicit === "help") assert.fail("expected parsed CLI options");
  assert.equal(explicit.model, "claude-opus-5");
  assert.equal(explicit.anthropicUrl, "https://example.test");
});

test("an unknown proposal backend is rejected", () => {
  assert.throws(
    () => parseCliArgs(["spec.json", "--proposal", "openai"]),
    (error: unknown) =>
      error instanceof CliArgumentError &&
      /must be catalog, ollama, anthropic, or grammar-smc/.test(error.message),
  );
});

test("the finite grammar SMC control parses its exact-target options", () => {
  const defaults = parseCliArgs(["spec.json", "--proposal", "grammar-smc"]);
  if (defaults === "help") assert.fail("expected parsed CLI options");
  assert.equal(defaults.proposal, "grammar-smc");
  assert.equal(defaults.grammarMaxCost, 5);
  assert.equal(defaults.grammarLimit, 100_000);
  assert.equal(defaults.betaMax, 1);
  assert.equal(defaults.movesPerStage, 1);

  const overridden = parseCliArgs([
    "spec.json",
    "--proposal",
    "grammar-smc",
    "--grammar-max-cost",
    "7",
    "--grammar-limit",
    "250000",
    "--beta-max",
    "2.5",
    "--moves-per-stage",
    "3",
  ]);
  if (overridden === "help") assert.fail("expected parsed CLI options");
  assert.equal(overridden.grammarMaxCost, 7);
  assert.equal(overridden.grammarLimit, 250_000);
  assert.equal(overridden.betaMax, 2.5);
  assert.equal(overridden.movesPerStage, 3);

  assert.throws(
    () => parseCliArgs(["spec.json", "--grammar-max-cost", "0"]),
    /must be positive/,
  );
  assert.throws(
    () => parseCliArgs(["spec.json", "--moves-per-stage", "-1"]),
    /must be nonnegative/,
  );
});
