/**
 * Importance sampling with the LLM itself as the proposal distribution,
 * with mathematically exact weights from stock Ollama.
 *
 * Construction (see CALIBRATION.md section "LLM proposal, exact weights"):
 * - The model samples UNCONSTRAINED (no grammar mask) with every sampler
 *   neutralized (temperature 1, top_k 0, top_p 1, min_p 0, repeat_penalty 1),
 *   so the returned per-token logprobs are exactly the sampling distribution.
 * - A draw is accepted only if its text is a designated serialization of a
 *   verifier-accepted program within the cost cap (canonical minified JSON,
 *   optionally with one trailing newline — a constant-size designated set per
 *   program). All other draws get weight zero.
 * - Self-normalized weights  w_i = exp( -beta*cost_i + logLik_i - log q(text_i) )
 *   then estimate the true posterior: the unknown constants — P(accept) and
 *   the designated-set size — are identical across draws and cancel.
 *
 * The estimator is validated against the exact enumerated posterior.
 *
 * Usage:
 *   npm run llm-is -- <config.json> --cost-cap N [--model qwen3-coder:30b-a3b-q8_0]
 *     [--draws 200] [--beta X] [--top K] [--ollama-url http://localhost:11434]
 */
import { decodeProgram } from "../src/shell/ast/decode.js";
import { jsonStringify, programToJsonValue, renderProgram } from "../src/shell/ast/render.js";
import { scoreCalibrated } from "../src/shell/scoring/calibrated.js";
import { computeExactPosterior, formatProbability, parseHarnessArgs } from "./posterior-lib.js";

const argv = process.argv.slice(2);
const args = parseHarnessArgs(argv);
const flag = (name: string, fallback: string): string => {
  const index = argv.indexOf(name);
  return index === -1 ? fallback : argv[index + 1]!;
};
const model = flag("--model", "qwen3-coder:30b-a3b-q8_0");
const draws = Number(flag("--draws", "200"));
const baseUrl = flag("--ollama-url", "http://localhost:11434").replace(/\/$/, "");

console.log(`[llm-is] computing exact posterior for ground truth (cap ${args.costCap})...`);
const exact = computeExactPosterior(args);
console.log(
  `[llm-is] support ${exact.programCount}; exact-solve mass ${formatProbability(exact.exactSolveMass)}; entropy ${exact.entropy.toFixed(3)} nats`,
);
const exactSolvers = new Set(exact.entries.filter((entry) => entry.exact).map((entry) => entry.rendered));

const examplesText = args.config.examples
  .map((example) => `${renderExample(example.input)} -> ${renderExample(example.output)}`)
  .join("; ");
function renderExample(value: unknown): string {
  return JSON.stringify(JSON.parse(jsonStringify(toPlain(value))));
}
function toPlain(value: unknown): unknown {
  const v = value as { kind: string; intValue?: bigint; boolValue?: boolean; intListValue?: unknown; boolListValue?: unknown };
  if (v.kind === "IntValue") return v.intValue!.toString();
  if (v.kind === "BoolValue") return v.boolValue;
  const items: unknown[] = [];
  let tail = (v.kind === "IntListValue" ? v.intListValue : v.boolListValue) as {
    kind: string;
    head?: unknown;
    tail?: unknown;
  };
  while (tail.kind === "IntCons" || tail.kind === "BoolCons") {
    items.push(typeof tail.head === "bigint" ? tail.head.toString() : tail.head);
    tail = tail.tail as typeof tail;
  }
  return items;
}

const PROMPT = [
  `You are a program synthesizer. Output ONLY minified JSON on a single line: no code fences, no commentary, no whitespace, no trailing newline.`,
  `Produce one complete program AST for a ${args.config.inputType} -> ${args.config.outputType} function matching ALL examples.`,
  `Root: {"kind":"ExpressionProgram","body":E} | {"kind":"MapProgram","mapper":E} | {"kind":"FoldRightProgram","initial":E,"reducer":E}.`,
  `E: {"kind":"Input"} | {"kind":"Item"} | {"kind":"Accumulator"} | {"kind":"IntLiteral","intValue":"<decimal>"} | {"kind":"BoolLiteral","boolValue":true|false} | {"kind":"EmptyIntList"} | {"kind":"EmptyBoolList"} | {"kind":"PrependInt","head":E,"tail":E} | {"kind":"PrependBool","head":E,"tail":E} | {"kind":"Not","operand":E} | {"kind":"Add","left":E,"right":E} | {"kind":"Subtract","left":E,"right":E} | {"kind":"Multiply","left":E,"right":E} | {"kind":"LessThan","left":E,"right":E} | {"kind":"EqualInt","left":E,"right":E} | {"kind":"And","left":E,"right":E} | {"kind":"IfThenElse","condition":E,"thenExpr":E,"elseExpr":E}.`,
  `IntLiteral values must be one of: ${args.config.integerConstants.map((c) => c.toString()).join(", ")}.`,
  `Item is only valid inside a MapProgram mapper or FoldRightProgram reducer; Accumulator only inside a reducer; never use Input inside mapper/initial/reducer.`,
  `Keep the program small (total nodes <= ${args.costCap}).`,
  `Examples: ${examplesText}`,
].join("\n");

interface OllamaChatResponse {
  readonly message?: { readonly content?: string };
  readonly logprobs?: readonly { readonly token: string; readonly logprob: number }[];
  readonly done_reason?: string;
}

async function drawOnce(): Promise<{
  readonly text: string;
  readonly sumLogProb: number;
  readonly tokenCount: number;
  readonly doneReason: string;
}> {
  const response = await fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model,
      stream: false,
      logprobs: true,
      messages: [{ role: "user", content: PROMPT }],
      options: {
        temperature: 1,
        top_k: 0,
        top_p: 1,
        min_p: 0,
        repeat_penalty: 1,
        num_predict: 700,
        seed: -1,
      },
    }),
  });
  if (!response.ok) throw new Error(`Ollama HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`);
  const body = (await response.json()) as OllamaChatResponse;
  const text = body.message?.content ?? "";
  const logprobs = body.logprobs ?? [];
  const sumLogProb = logprobs.reduce((sum, entry) => sum + entry.logprob, 0);
  return { text, sumLogProb, tokenCount: logprobs.length, doneReason: body.done_reason ?? "?" };
}

interface AcceptedDraw {
  readonly rendered: string;
  readonly logWeight: number;
  readonly cost: number;
}

function tryAccept(text: string, sumLogProb: number): AcceptedDraw | { readonly reject: string } {
  // Designated serializations per program: canonical minified JSON, with at
  // most one trailing newline (constant set size for every program).
  const stripped = text.endsWith("\n") ? text.slice(0, -1) : text;
  let parsed: unknown;
  try {
    parsed = JSON.parse(stripped);
  } catch {
    return { reject: "not JSON" };
  }
  let program;
  try {
    program = decodeProgram(parsed, {
      integerConstants: args.config.integerConstants,
      maxDepth: args.config.maxDepth,
      maxNodes: args.config.maxNodes,
    });
  } catch (error) {
    return { reject: `decode: ${error instanceof Error ? error.message.slice(0, 80) : "?"}` };
  }
  const canonical = jsonStringify(programToJsonValue(program));
  if (stripped !== canonical) return { reject: "non-canonical serialization" };
  const score = scoreCalibrated(program, {
    inputType: args.config.inputType,
    outputType: args.config.outputType,
    examples: args.config.examples,
    beta: args.beta,
    noise: args.noise,
  });
  if ("rejected" in score) return { reject: score.rejected };
  if (score.cost > args.costCap) return { reject: `cost ${score.cost} above cap ${args.costCap}` };
  return {
    rendered: renderProgram(program, args.config.inputType),
    logWeight: score.logPriorUnnormalized + score.logLikelihood - sumLogProb,
    cost: score.cost,
  };
}

console.log(`[llm-is] proposal: ${model} at temperature 1, all truncation samplers OFF, no grammar mask`);
console.log(
  `[llm-is] CAVEAT (v1): Ollama omits the EOS token from logprobs (probe: n_logprobs = eval_count - 1 on stop),`,
);
console.log(
  `[llm-is]   so weights omit the termination factor p(EOS|t). It is constant per program text; the residual`,
);
console.log(
  `[llm-is]   bias is only the cross-program ratio of EOS probabilities (~1 for compliant completions).`,
);
console.log(`[llm-is] drawing ${draws} iid proposals...`);

const accepted: AcceptedDraw[] = [];
const rejectionCounts = new Map<string, number>();
let doneReasonNonStop = 0;
for (let draw = 1; draw <= draws; draw += 1) {
  const { text, sumLogProb, doneReason } = await drawOnce();
  if (doneReason !== "stop") doneReasonNonStop += 1;
  const outcome = tryAccept(text, sumLogProb);
  if ("reject" in outcome) {
    rejectionCounts.set(outcome.reject, (rejectionCounts.get(outcome.reject) ?? 0) + 1);
  } else {
    accepted.push(outcome);
  }
  if (draw % 25 === 0) {
    console.log(`  ${draw}/${draws} drawn; accepted ${accepted.length} (${((100 * accepted.length) / draw).toFixed(1)}%)`);
  }
}

if (accepted.length === 0) {
  console.log("[llm-is] no draws accepted; cannot form estimates. Rejection reasons:");
  for (const [reason, count] of rejectionCounts) console.log(`  ${count}x ${reason}`);
  process.exit(1);
}

const maxLogWeight = Math.max(...accepted.map((entry) => entry.logWeight));
const weights = accepted.map((entry) => Math.exp(entry.logWeight - maxLogWeight));
const weightSum = weights.reduce((sum, weight) => sum + weight, 0);
const effectiveSampleSize = (weightSum * weightSum) / weights.reduce((sum, weight) => sum + weight * weight, 0);

const massByRendering = new Map<string, number>();
let exactMass = 0;
accepted.forEach((entry, index) => {
  const normalized = weights[index]! / weightSum;
  massByRendering.set(entry.rendered, (massByRendering.get(entry.rendered) ?? 0) + normalized);
  if (exactSolvers.has(entry.rendered)) exactMass += normalized;
});

let totalVariation = 0;
for (const [rendered, truth] of exact.probabilityByRendering) {
  totalVariation += Math.abs((massByRendering.get(rendered) ?? 0) - truth);
}
for (const [rendered, estimate] of massByRendering) {
  if (!exact.probabilityByRendering.has(rendered)) totalVariation += estimate; // mass outside exact support (should be zero)
}
totalVariation /= 2;

console.log(`\n[llm-is] accepted ${accepted.length}/${draws} draws (${((100 * accepted.length) / draws).toFixed(1)}%); non-stop done_reason: ${doneReasonNonStop}`);
console.log("[llm-is] rejection breakdown:");
for (const [reason, count] of [...rejectionCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6)) {
  console.log(`  ${String(count).padStart(4)}x ${reason}`);
}
console.log(`[llm-is] ESS = ${effectiveSampleSize.toFixed(1)} of ${accepted.length} accepted (${((100 * effectiveSampleSize) / accepted.length).toFixed(1)}%)`);
console.log(`[llm-is] exact-solve mass: estimate ${formatProbability(exactMass)} vs truth ${formatProbability(exact.exactSolveMass)}`);
console.log(`[llm-is] total-variation distance to exact posterior: ${totalVariation.toFixed(6)}`);
console.log(`\n[llm-is] top ${args.top} exact-posterior programs vs LLM-IS estimates:`);
for (const entry of exact.entries.slice(0, args.top)) {
  console.log(
    `  true=${formatProbability(entry.probability).padStart(12)}  est=${formatProbability(massByRendering.get(entry.rendered) ?? 0).padStart(12)}  ${entry.exact ? "EXACT" : "     "}  ${entry.rendered.slice(0, 90)}`,
  );
}
console.log(`\n[llm-is] top LLM-IS atoms not shown above:`);
const shown = new Set(exact.entries.slice(0, args.top).map((entry) => entry.rendered));
for (const [rendered, estimate] of [...massByRendering.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5)) {
  if (shown.has(rendered)) continue;
  console.log(`  est=${formatProbability(estimate).padStart(12)}  true=${formatProbability(exact.probabilityByRendering.get(rendered) ?? 0).padStart(12)}  ${rendered.slice(0, 90)}`);
}
