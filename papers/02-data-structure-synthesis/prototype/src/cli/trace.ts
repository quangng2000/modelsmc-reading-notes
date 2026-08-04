import {
  renderExpression,
  renderPrimitiveValue,
  renderType,
  type PrimitiveValue,
} from "../ast.js";
import type {
  FamilyDeduction,
  IOExample,
  SynthesisEvent,
  SynthesisSignature,
} from "../synthesis/index.js";

export function reportRequest(
  examples: readonly IOExample[],
  signature?: SynthesisSignature,
): void {
  writeTrace(
    `loaded ${examples.length} example${examples.length === 1 ? "" : "s"}`,
  );
  if (signature !== undefined) {
    writeTrace(
      `signature: ${renderType(signature.inputType)} -> ${renderType(signature.outputType)}`,
    );
  }
  examples.forEach((example, index) => {
    writeTrace(
      `example ${index + 1}: ${renderList(example.input)} -> ${renderOutput(example.output)}`,
    );
  });
}

export function reportSynthesisEvent(event: SynthesisEvent): void {
  switch (event.type) {
    case "search-start":
      writeTrace(
        `search started; possible families: ${event.families.join(", ")}`,
      );
      return;
    case "family-refuted":
      writeTrace(`family ${event.family}: refuted — ${event.reason}`);
      return;
    case "family-viable":
      writeTrace(
        `family ${event.family}: viable skeleton ${renderExpression(event.skeleton)} (cost ${event.cost})`,
      );
      reportDeduction(event.deduction);
      return;
    case "candidate-tested": {
      const disposition =
        event.disposition === "rejected-by-deduction"
          ? "rejected by deduction"
          : event.disposition === "rejected-by-examples"
            ? "rejected by full examples"
            : "accepted";
      const reason =
        event.disposition === "rejected-by-deduction"
          ? ` — ${event.reason}`
          : "";
      writeTrace(
        `candidate ${event.number}: ${event.family}, cost ${event.cost}, ${renderExpression(event.program)} -> ${disposition}${reason}`,
      );
      return;
    }
    case "search-finished":
      writeTrace(
        `search finished: ${event.outcome} after ${event.candidatesTested} tested candidate${event.candidatesTested === 1 ? "" : "s"}`,
      );
  }
}

function reportDeduction(deduction: FamilyDeduction): void {
  switch (deduction.kind) {
    case "map":
      writeTrace(
        `deduction subproblem type: ?f: ${renderType(deduction.inputType)} -> ${renderType(deduction.outputType)}`,
      );
      writeTrace(
        deduction.examples.length === 0
          ? "deduction for ?f: no element examples inferred"
          : `deduction for ?f: ${deduction.examples
              .map(
                ({ input, output }) =>
                  `${renderPrimitiveValue(input)} -> ${renderPrimitiveValue(output)}`,
              )
              .join(", ")}`,
      );
      return;
    case "filter":
      writeTrace(
        `deduction subproblem type: ?p: ${renderType(deduction.elementType)} -> bool`,
      );
      writeTrace(
        deduction.examples.length === 0
          ? "deduction for ?p: no predicate examples inferred"
          : `deduction for ?p: ${deduction.examples
              .map(
                ({ input, output }) =>
                  `${renderPrimitiveValue(input)} -> ${output}`,
              )
              .join(", ")}`,
      );
      return;
    case "fold": {
      writeTrace(
        `deduction subproblem types: ?f: ${renderType(deduction.accumulatorType)} -> ${renderType(deduction.elementType)} -> ${renderType(deduction.accumulatorType)}, ?init: ${renderType(deduction.accumulatorType)}`,
      );
      const init =
        deduction.init === undefined
          ? deduction.initCandidates.length === 0
            ? "not fixed; no candidates"
            : `not fixed; candidates = ${deduction.initCandidates.map(renderPrimitiveValue).join(", ")}`
          : renderPrimitiveValue(deduction.init);
      writeTrace(`deduction for ?init: ${init}`);
      writeTrace(
        deduction.steps.length === 0
          ? "deduction for reducer ?f: no steps inferred"
          : `deduction for reducer ?f: ${deduction.steps
              .map(
                ({ accumulator, element, output }) =>
                  `(${renderPrimitiveValue(accumulator)}, ${renderPrimitiveValue(element)}) -> ${renderPrimitiveValue(output)}`,
              )
              .join(", ")}`,
      );
    }
  }
}

function renderOutput(output: IOExample["output"]): string {
  return isPrimitive(output)
    ? renderPrimitiveValue(output)
    : renderList(output);
}

function renderList(values: readonly PrimitiveValue[]): string {
  return `[${values.map(renderPrimitiveValue).join(", ")}]`;
}

function isPrimitive(
  output: IOExample["output"],
): output is PrimitiveValue {
  return (
    typeof output === "number" ||
    typeof output === "boolean" ||
    typeof output === "string"
  );
}

function writeTrace(message: string): void {
  process.stdout.write(`[trace] ${message}\n`);
}
