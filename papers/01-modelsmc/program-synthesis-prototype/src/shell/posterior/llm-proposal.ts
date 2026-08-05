import { decodeProgram } from "../ast/decode.js";
import { jsonStringify, programToJsonValue, renderProgram } from "../ast/render.js";
import type { Program, RuntimeValue, StaticType } from "../../core/language.verify.js";
import { scoreCalibrated } from "../scoring/calibrated.js";
import type { NoiseModel } from "../scoring/emission.js";

/**
 * The LLM as a proposal distribution with evaluable density (v1, stock Ollama).
 *
 * Draws are unconstrained (no grammar mask) with every sampler neutralized, so
 * the returned per-token logprobs are the sampling distribution; only texts in
 * a program's designated serialization set (canonical minified JSON, optional
 * single trailing newline) are accepted. Known v1 gap: the EOS termination
 * factor is absent from Ollama logprobs (constant per text; cancels within a
 * program, residual cross-program bias ~1). See CALIBRATION.md.
 */

export interface ProposalTask {
  readonly inputType: StaticType;
  readonly outputType: StaticType;
  readonly examples: readonly { readonly input: RuntimeValue; readonly output: RuntimeValue }[];
  readonly integerConstants: readonly bigint[];
  readonly maxDepth: number;
  readonly maxNodes: number;
  readonly costCap: number;
  readonly beta: number;
  readonly noise: NoiseModel;
}

export interface AcceptedProposal {
  readonly program: Program;
  readonly rendered: string;
  readonly cost: number;
  readonly logLikelihood: number;
  /** -beta * cost (unnormalized log prior). */
  readonly logPriorUnnormalized: number;
  /** log q of the emitted text under the neutralized sampler (content tokens). */
  readonly logProposal: number;
}

export type ProposalOutcome = AcceptedProposal | { readonly rejected: string };

function renderValuePlain(value: RuntimeValue): string {
  if (value.kind === "IntValue") return JSON.stringify(value.intValue.toString());
  if (value.kind === "BoolValue") return JSON.stringify(value.boolValue);
  const items: string[] = [];
  if (value.kind === "IntListValue") {
    let tail = value.intListValue;
    while (tail.kind === "IntCons") {
      items.push(JSON.stringify(tail.head.toString()));
      tail = tail.tail;
    }
  } else {
    let tail = value.boolListValue;
    while (tail.kind === "BoolCons") {
      items.push(JSON.stringify(tail.head));
      tail = tail.tail;
    }
  }
  return `[${items.join(",")}]`;
}

export function proposalPrompt(task: ProposalTask): string {
  const examplesText = task.examples
    .map((example) => `${renderValuePlain(example.input)} -> ${renderValuePlain(example.output)}`)
    .join("; ");
  return [
    `You are a program synthesizer. Output ONLY minified JSON on a single line: no code fences, no commentary, no whitespace, no trailing newline.`,
    `Produce one complete program AST for a ${task.inputType} -> ${task.outputType} function matching ALL examples.`,
    `Root: {"kind":"ExpressionProgram","body":E} | {"kind":"MapProgram","mapper":E} | {"kind":"FoldRightProgram","initial":E,"reducer":E}.`,
    `E: {"kind":"Input"} | {"kind":"Item"} | {"kind":"Accumulator"} | {"kind":"IntLiteral","intValue":"<decimal>"} | {"kind":"BoolLiteral","boolValue":true|false} | {"kind":"EmptyIntList"} | {"kind":"EmptyBoolList"} | {"kind":"PrependInt","head":E,"tail":E} | {"kind":"PrependBool","head":E,"tail":E} | {"kind":"Not","operand":E} | {"kind":"Add","left":E,"right":E} | {"kind":"Subtract","left":E,"right":E} | {"kind":"Multiply","left":E,"right":E} | {"kind":"LessThan","left":E,"right":E} | {"kind":"EqualInt","left":E,"right":E} | {"kind":"And","left":E,"right":E} | {"kind":"IfThenElse","condition":E,"thenExpr":E,"elseExpr":E}.`,
    `IntLiteral values must be one of: ${task.integerConstants.map((constant) => constant.toString()).join(", ")}.`,
    `Item is only valid inside a MapProgram mapper or FoldRightProgram reducer; Accumulator only inside a reducer; never use Input inside mapper/initial/reducer.`,
    `Keep the program small (total nodes <= ${task.costCap}).`,
    `Examples: ${examplesText}`,
  ].join("\n");
}

export interface RawDraw {
  readonly text: string;
  readonly sumLogProb: number;
  readonly doneReason: string;
}

/** One unconstrained draw from Ollama with neutralized samplers. */
export function createOllamaDrawer(options: {
  readonly model: string;
  readonly baseUrl?: string;
  readonly prompt: string;
  readonly numPredict?: number;
  readonly requester?: typeof fetch;
}): () => Promise<RawDraw> {
  const baseUrl = (options.baseUrl ?? "http://localhost:11434").replace(/\/$/, "");
  const requester = options.requester ?? fetch;
  return async () => {
    const response = await requester(`${baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: options.model,
        stream: false,
        logprobs: true,
        messages: [{ role: "user", content: options.prompt }],
        options: {
          temperature: 1,
          top_k: 0,
          top_p: 1,
          min_p: 0,
          repeat_penalty: 1,
          num_predict: options.numPredict ?? 700,
          seed: -1,
        },
      }),
    });
    if (!response.ok) {
      throw new Error(`Ollama HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`);
    }
    const body = (await response.json()) as {
      message?: { content?: string };
      logprobs?: readonly { logprob: number }[];
      done_reason?: string;
    };
    return {
      text: body.message?.content ?? "",
      sumLogProb: (body.logprobs ?? []).reduce((sum, entry) => sum + entry.logprob, 0),
      doneReason: body.done_reason ?? "?",
    };
  };
}

/** Accept a raw draw iff its text is a designated serialization of an in-support program. */
export function evaluateDraw(draw: RawDraw, task: ProposalTask): ProposalOutcome {
  const stripped = draw.text.endsWith("\n") ? draw.text.slice(0, -1) : draw.text;
  let parsed: unknown;
  try {
    parsed = JSON.parse(stripped);
  } catch {
    return { rejected: "not JSON" };
  }
  let program: Program;
  try {
    program = decodeProgram(parsed, {
      integerConstants: task.integerConstants,
      maxDepth: task.maxDepth,
      maxNodes: task.maxNodes,
    });
  } catch (error) {
    return { rejected: `decode: ${error instanceof Error ? error.message.slice(0, 80) : "?"}` };
  }
  if (stripped !== jsonStringify(programToJsonValue(program))) {
    return { rejected: "non-canonical serialization" };
  }
  const score = scoreCalibrated(program, {
    inputType: task.inputType,
    outputType: task.outputType,
    examples: task.examples,
    beta: task.beta,
    noise: task.noise,
  });
  if ("rejected" in score) return { rejected: score.rejected };
  if (score.cost > task.costCap) return { rejected: `cost ${score.cost} above cap ${task.costCap}` };
  return {
    program,
    rendered: renderProgram(program, task.inputType),
    cost: score.cost,
    logLikelihood: score.logLikelihood,
    logPriorUnnormalized: score.logPriorUnnormalized,
    logProposal: draw.sumLogProb,
  };
}

/**
 * Draw until acceptance. Conditioning on acceptance rescales the proposal by
 * the constant 1/P(accept), which cancels in self-normalized weights and in
 * MH ratios where both sides use the same conditioned proposal.
 */
export async function drawAccepted(
  drawer: () => Promise<RawDraw>,
  task: ProposalTask,
  onReject?: (reason: string) => void,
  maxAttempts = 200,
): Promise<AcceptedProposal> {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const outcome = evaluateDraw(await drawer(), task);
    if ("rejected" in outcome) {
      onReject?.(outcome.rejected);
      continue;
    }
    return outcome;
  }
  throw new Error(`no accepted proposal in ${maxAttempts} attempts`);
}
