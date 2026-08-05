import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

// Tests run post-build from dist/test, so the built CLI sits at dist/src.
const CLI_PATH = fileURLToPath(new URL("../src/cli.js", import.meta.url));
const DOCUMENTED_EXAMPLE_PATH = resolve(
  "examples",
  "keep-positive.json",
);
const PAPER_EXAMPLE_PATH = resolve("examples", "paper-map-example.json");
const PAPER_MULTIPLE_EXAMPLE_PATH = resolve(
  "examples",
  "paper-multiple-map-example.json",
);
const MULTIPLE_FOLD_EXAMPLE_PATH = resolve(
  "examples",
  "multiple-sum-fold.json",
);

interface CliRun {
  readonly status: number;
  readonly stdout: string;
  readonly stderr: string;
}

function runCli(args: readonly string[]): CliRun {
  try {
    const stdout = execFileSync(process.execPath, [CLI_PATH, ...args], {
      encoding: "utf8",
    });
    return { status: 0, stdout, stderr: "" };
  } catch (error) {
    const failure = error as {
      readonly status?: number | null;
      readonly stdout?: string;
      readonly stderr?: string;
    };
    return {
      status: failure.status ?? -1,
      stdout: failure.stdout ?? "",
      stderr: failure.stderr ?? "",
    };
  }
}

function assertCompleteCandidateTrace(
  run: CliRun,
  expectedCount: number,
): readonly string[] {
  const candidateLines = run.stdout
    .split("\n")
    .filter((line) => line.startsWith("[trace] candidate "));

  assert.equal(candidateLines.length, expectedCount);
  candidateLines.forEach((line, index) => {
    assert.ok(line.startsWith(`[trace] candidate ${index + 1}:`));
  });
  assert.ok(
    candidateLines.slice(0, -1).every((line) => line.includes("rejected")),
  );
  assert.ok(candidateLines.at(-1)?.endsWith("accepted"));
  return candidateLines;
}

test("synthesizes the documented program from the included JSON", () => {
  const run = runCli([DOCUMENTED_EXAMPLE_PATH]);

  assert.equal(run.status, 0);
  assert.ok(!run.stdout.includes("[trace]"));
  assert.ok(run.stdout.includes("family:  filter"));
  assert.ok(run.stdout.includes("filter("));
  assert.ok(run.stdout.includes("deduction:"));
});

test("prints every synthesis stage and tested candidate with --trace", () => {
  const run = runCli(["--trace", DOCUMENTED_EXAMPLE_PATH]);

  assert.equal(run.status, 0);
  assert.ok(run.stdout.includes("[trace] loaded 1 example"));
  assert.ok(run.stdout.includes("[trace] example 1:"));
  assert.ok(run.stdout.includes("possible families: map, filter"));
  assert.ok(run.stdout.includes("family map: refuted"));
  assert.ok(run.stdout.includes("family filter: viable skeleton"));
  assert.ok(
    run.stdout.includes("deduction subproblem type: ?p: int -> bool"),
  );
  assert.ok(
    run.stdout.includes(
      "candidate 1: filter, cost 4, (xs: list<int>) => filter((x: int) => true, xs) -> rejected by deduction — inferred sub-example ?p(-2): expected false, got true",
    ),
  );
  assertCompleteCandidateTrace(run, 13);
  assert.ok(
    run.stdout.includes("search finished: synthesized after 13 tested candidates"),
  );
});

test("traces the paper's map example through every tested candidate", () => {
  const run = runCli(["--trace", PAPER_EXAMPLE_PATH]);

  assert.equal(run.status, 0);
  assert.ok(run.stdout.includes("[trace] example 1: [1, 2] -> [3, 4]"));
  assert.ok(run.stdout.includes("possible families: map, filter"));
  assert.ok(run.stdout.includes("family map: viable skeleton"));
  assert.ok(run.stdout.includes("deduction subproblem type: ?f: int -> int"));
  assert.ok(run.stdout.includes("deduction for ?f: 1 -> 3, 2 -> 4"));
  assert.ok(run.stdout.includes("family filter: refuted"));

  const candidateLines = assertCompleteCandidateTrace(run, 11);
  assert.ok(
    candidateLines[0]?.includes(
      "rejected by deduction — inferred sub-example ?f(1): expected 3, got 1",
    ),
  );
  assert.ok(candidateLines.at(-1)?.includes("(x + 2)"));
  assert.ok(
    run.stdout.includes("search finished: synthesized after 11 tested candidates"),
  );
  assert.ok(
    run.stdout.includes(
      "program: (xs: list<int>) => map((x: int) => (x + 2), xs)",
    ),
  );
});

test("traces the paper's multiple-example map deduction", () => {
  const run = runCli(["--trace", PAPER_MULTIPLE_EXAMPLE_PATH]);

  assert.equal(run.status, 0);
  assert.ok(run.stdout.includes("[trace] loaded 2 examples"));
  assert.ok(run.stdout.includes("[trace] example 1: [1, 2] -> [2, 3]"));
  assert.ok(run.stdout.includes("[trace] example 2: [2, 4] -> [3, 5]"));
  assert.ok(run.stdout.includes("family map: viable skeleton"));
  assert.ok(run.stdout.includes("deduction for ?f: 1 -> 2, 2 -> 3, 4 -> 5"));
  assert.ok(run.stdout.includes("family filter: refuted"));

  const candidateLines = assertCompleteCandidateTrace(run, 10);
  assert.ok(candidateLines.at(-1)?.includes("(x + 1)"));
  assert.ok(
    run.stdout.includes("search finished: synthesized after 10 tested candidates"),
  );
});

test("traces a multi-example fold specification through every candidate", () => {
  const run = runCli(["--trace", MULTIPLE_FOLD_EXAMPLE_PATH]);

  assert.equal(run.status, 0);
  assert.ok(run.stdout.includes("[trace] loaded 5 examples"));
  assert.ok(run.stdout.includes("[trace] example 1: [] -> 0"));
  assert.ok(run.stdout.includes("[trace] example 5: [-2, 5, 1] -> 4"));
  assert.ok(run.stdout.includes("possible families: fold"));
  assert.ok(run.stdout.includes("family fold: viable skeleton"));
  assert.ok(
    run.stdout.includes(
      "deduction subproblem types: ?f: int -> int -> int, ?init: int",
    ),
  );
  assert.ok(run.stdout.includes("deduction for ?init: 0"));
  assert.ok(
    run.stdout.includes(
      "deduction for reducer ?f: (0, 1) -> 1, (1, 2) -> 3, (3, 3) -> 6",
    ),
  );

  const candidateLines = assertCompleteCandidateTrace(run, 8);
  assert.ok(
    candidateLines[0]?.includes(
      "rejected by deduction — inferred reducer sub-example ?f(0, 1): expected 1, got 0",
    ),
  );
  assert.ok(
    candidateLines[1]?.includes(
      "rejected by deduction — inferred reducer sub-example ?f(1, 2): expected 3, got 2",
    ),
  );
  assert.ok(candidateLines.at(-1)?.includes("(acc + x)"));
  assert.ok(
    run.stdout.includes("search finished: synthesized after 8 tested candidates"),
  );
});

test("traces evaluation errors while checking inferred sub-examples", () => {
  const request = JSON.stringify({
    examples: [{ input: [0], output: [1] }],
    maxCost: 2,
    constants: [0],
  });
  const run = runCli(["--trace", "-e", request]);

  assert.equal(run.status, 1);
  assert.ok(
    run.stdout.includes(
      "inferred sub-example ?f(0): expected 1; evaluation error: Modulo by zero.",
    ),
  );
});

test("rejects invalid input with a one-line stderr error and exit 2", () => {
  const empty = runCli(["-e", '{"examples": []}']);
  assert.equal(empty.status, 2);
  assert.match(empty.stderr, /nonempty array/);

  const malformed = runCli(["-e", "not json"]);
  assert.equal(malformed.status, 2);
  assert.match(malformed.stderr, /not valid JSON/);
});

test("numbers legacy validation errors from one", () => {
  const cases = [
    [
      '{"examples": [null]}',
      "error: example 1 must be an object\n",
    ],
    [
      '{"examples": [{"input": "no", "output": [1]}]}',
      "error: example 1 input must be an array of safe integers\n",
    ],
    [
      '{"examples": [{"input": [1], "output": "no"}]}',
      "error: example 1 output must be a safe integer or an array of safe integers\n",
    ],
  ] as const;

  for (const [request, expectedError] of cases) {
    const run = runCli(["-e", request]);
    assert.equal(run.status, 2);
    assert.equal(run.stdout, "");
    assert.equal(run.stderr, expectedError);
  }
});

test("exits 1 when every family is refuted", () => {
  const run = runCli(["-e", '{"examples": [{"input": [1, 1], "output": [2, 3]}]}']);

  assert.equal(run.status, 1);
  assert.ok(run.stdout.includes("refuted"));
});
