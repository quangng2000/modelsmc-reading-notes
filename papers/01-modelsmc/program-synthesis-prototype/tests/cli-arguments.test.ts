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
      /must be catalog, ollama, or anthropic/.test(error.message),
  );
});
