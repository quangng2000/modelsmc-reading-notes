import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

// Tests run post-build from dist/test, so the built CLI sits at dist/src.
const CLI_PATH = fileURLToPath(new URL("../src/cli.js", import.meta.url));

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

function runFixture(name: string): CliRun {
  return runCli(["--trace", resolve("examples", name)]);
}

function runInline(request: Readonly<Record<string, unknown>>): CliRun {
  return runCli(["-e", JSON.stringify(request)]);
}

function assertTraceLine(run: CliRun, line: string): void {
  assert.ok(
    run.stdout.split("\n").includes(`[trace] ${line}`),
    `missing exact trace line: ${line}`,
  );
}

function assertProgramLine(run: CliRun, program: string): void {
  assert.ok(
    run.stdout.split("\n").includes(`program: ${program}`),
    `missing exact program line: ${program}`,
  );
}

test("synthesizes a string-equality map with declared bool output", () => {
  const run = runFixture("string-is-yes-map.json");

  assert.equal(run.status, 0);
  assert.equal(run.stderr, "");
  assertTraceLine(run, "signature: list<string> -> list<bool>");
  assertTraceLine(run, "search started; possible families: map");
  assertTraceLine(run, "deduction subproblem type: ?f: string -> bool");
  assertTraceLine(
    run,
    'deduction for ?f: "yes" -> true, "no" -> false, "maybe" -> false',
  );
  assertProgramLine(
    run,
    '(xs: list<string>) => map((x: string) => (x == "yes"), xs)',
  );
});

test("synthesizes string lengths with an exact string-to-int hole type", () => {
  const run = runFixture("string-lengths-map.json");

  assert.equal(run.status, 0);
  assert.equal(run.stderr, "");
  assertTraceLine(run, "signature: list<string> -> list<int>");
  assertTraceLine(run, "search started; possible families: map");
  assertTraceLine(run, "deduction subproblem type: ?f: string -> int");
  assertProgramLine(
    run,
    "(xs: list<string>) => map((x: string) => length(x), xs)",
  );
});

test("synthesizes boolean any as a bool accumulator fold", () => {
  const run = runFixture("boolean-any-fold.json");

  assert.equal(run.status, 0);
  assert.equal(run.stderr, "");
  assertTraceLine(run, "signature: list<bool> -> bool");
  assertTraceLine(run, "search started; possible families: fold");
  assertTraceLine(
    run,
    "deduction subproblem types: ?f: bool -> bool -> bool, ?init: bool",
  );
  assertTraceLine(run, "deduction for ?init: false");
  assertProgramLine(
    run,
    "(xs: list<bool>) => foldl((acc: bool) => (x: bool) => (acc || x), false, xs)",
  );
});

test("synthesizes string concatenation as a string accumulator fold", () => {
  const run = runFixture("concatenate-strings-fold.json");

  assert.equal(run.status, 0);
  assert.equal(run.stderr, "");
  assertTraceLine(run, "signature: list<string> -> string");
  assertTraceLine(run, "search started; possible families: fold");
  assertTraceLine(
    run,
    "deduction subproblem types: ?f: string -> string -> string, ?init: string",
  );
  assertTraceLine(run, 'deduction for ?init: ""');
  assertProgramLine(
    run,
    '(xs: list<string>) => foldl((acc: string) => (x: string) => (acc ++ x), "", xs)',
  );
});

test("numbers non-object typed examples from one", () => {
  const run = runInline({
    inputType: "list<string>",
    outputType: "list<bool>",
    examples: [null],
  });

  assert.equal(run.status, 2);
  assert.equal(run.stdout, "");
  assert.equal(run.stderr, "error: example 1 must be an object\n");
});

for (const [label, request] of [
  [
    "inputType without outputType",
    {
      inputType: "list<string>",
      examples: [{ input: ["yes"], output: [true] }],
    },
  ],
  [
    "outputType without inputType",
    {
      outputType: "list<bool>",
      examples: [{ input: ["yes"], output: [true] }],
    },
  ],
] as const) {
  test(`rejects ${label}`, () => {
    const run = runInline(request);

    assert.equal(run.status, 2);
    assert.equal(run.stdout, "");
    assert.equal(
      run.stderr,
      'error: "inputType" and "outputType" must be provided together\n',
    );
  });
}

for (const [label, request, expectedError] of [
  [
    "an input value that does not match the declared element type",
    {
      inputType: "list<string>",
      outputType: "list<bool>",
      examples: [{ input: [1], output: [true] }],
    },
    "error: example 1 input[0] must match string\n",
  ],
  [
    "an output value that does not match the declared element type",
    {
      inputType: "list<string>",
      outputType: "list<bool>",
      examples: [{ input: ["yes"], output: [1] }],
    },
    "error: example 1 output[0] must match bool\n",
  ],
] as const) {
  test(`rejects ${label}`, () => {
    const run = runInline(request);

    assert.equal(run.status, 2);
    assert.equal(run.stdout, "");
    assert.equal(run.stderr, expectedError);
  });
}

for (const [label, request, expectedError] of [
  [
    "a nested input-list signature",
    {
      inputType: "list<list<int>>",
      outputType: "list<int>",
      examples: [{ input: [[1]], output: [1] }],
    },
    "error: inputType must be list<int>, list<bool>, or list<string>; received list<list<int>>\n",
  ],
  [
    "a nested output-list signature",
    {
      inputType: "list<int>",
      outputType: "list<list<int>>",
      examples: [{ input: [1], output: [[1]] }],
    },
    "error: outputType must be a primitive or list of primitives; received list<list<int>>\n",
  ],
] as const) {
  test(`rejects ${label}`, () => {
    const run = runInline(request);

    assert.equal(run.status, 2);
    assert.equal(run.stdout, "");
    assert.equal(run.stderr, expectedError);
  });
}
