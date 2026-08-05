import {
  INT,
  primitiveValueEquals,
  renderPrimitiveValue,
  type PrimitiveType,
  type PrimitiveValue,
} from "../ast.js";
import {
  validatePrimitiveList,
  validatePrimitiveValue,
} from "./types.js";

export interface FoldExample<
  Element extends PrimitiveValue = number,
  Accumulator extends PrimitiveValue = number,
> {
  readonly input: readonly Element[];
  readonly output: Accumulator;
}

export interface FoldStepExample<
  Element extends PrimitiveValue = number,
  Accumulator extends PrimitiveValue = number,
> {
  readonly accumulator: Accumulator;
  readonly element: Element;
  readonly output: Accumulator;
}

export type FoldDeductionResult<
  Element extends PrimitiveValue = number,
  Accumulator extends PrimitiveValue = number,
> =
  | {
      readonly kind: "inferred";
      readonly init: Accumulator | undefined;
      readonly steps: readonly FoldStepExample<Element, Accumulator>[];
    }
  | { readonly kind: "refuted"; readonly reason: string };

// Sound fold deduction: everything returned here is a consequence of the
// examples under a left fold with a pure reducer, so a candidate rejected by
// these facts can never satisfy the full examples.
export function deduceFoldExamples<
  Element extends PrimitiveValue = number,
  Accumulator extends PrimitiveValue = number,
>(
  examples: readonly FoldExample<Element, Accumulator>[],
  elementType: PrimitiveType = INT,
  accumulatorType: PrimitiveType = INT,
): FoldDeductionResult<Element, Accumulator> {
  // Rule 1: every value must agree with the declared primitive types.
  for (const example of examples) {
    validatePrimitiveList(example.input, elementType, "fold input");
    validatePrimitiveValue(example.output, accumulatorType, "fold output");
  }

  // Rule 2 (determinism): two examples with identical input lists must agree
  // on the output. This subsumes conflicting [] -> b examples. Duplicated
  // consistent examples collapse to one entry.
  const outputByInput = new Map<string, Accumulator>();
  const uniqueExamples: FoldExample<Element, Accumulator>[] = [];
  for (const example of examples) {
    const key = JSON.stringify(example.input);
    const priorOutput = outputByInput.get(key);
    if (priorOutput !== undefined) {
      if (!primitiveValueEquals(priorOutput, example.output)) {
        return {
          kind: "refuted",
          reason: `fold cannot send [${example.input.map(renderPrimitiveValue).join(", ")}] to both ${renderPrimitiveValue(priorOutput)} and ${renderPrimitiveValue(example.output)}`,
        };
      }
      continue;
    }
    outputByInput.set(key, example.output);
    uniqueExamples.push(example);
  }

  // Rule 3 (init inference): an example [] -> b fixes init = b, because a
  // left fold over the empty list returns its initial accumulator unchanged.
  const emptyExample = uniqueExamples.find(
    (example) => example.input.length === 0,
  );
  const init = emptyExample?.output;

  // Rule 4 ([x] -> b with a known init yields the step (init, x) -> b) is
  // exactly rule 5 applied to a virtual [] -> init example, so a known init
  // is represented as that virtual example before scanning pairs. Today init
  // only ever comes from an actual [] example (rule 3), which is already in
  // uniqueExamples, so the virtual example is added defensively for any
  // future init source.
  const peelable: readonly FoldExample<Element, Accumulator>[] =
    init !== undefined && emptyExample === undefined
      ? [{ input: [], output: init }, ...uniqueExamples]
      : uniqueExamples;

  // Rule 5 (one-element-extension peeling): for every pair of examples
  // (xs -> b1, ys -> b2) where ys is exactly xs plus one trailing element e,
  // a left fold reaches accumulator b1 after xs, so the reducer must send
  // (b1, e) to b2.
  // Rule 6: steps are deduped by (accumulator, element); two steps that
  // disagree on the output refute the fold family outright.
  const outputByStep = new Map<string, Accumulator>();
  const steps: FoldStepExample<Element, Accumulator>[] = [];
  for (const shorter of peelable) {
    for (const longer of peelable) {
      if (longer.input.length !== shorter.input.length + 1) {
        continue;
      }
      if (!isPrefix(shorter.input, longer.input)) {
        continue;
      }

      const element = longer.input[longer.input.length - 1];
      if (element === undefined) {
        throw new Error("A nonempty list was unexpectedly missing its last element.");
      }

      const stepKey = JSON.stringify([shorter.output, element]);
      const priorOutput = outputByStep.get(stepKey);
      if (priorOutput !== undefined) {
        if (!primitiveValueEquals(priorOutput, longer.output)) {
          return {
            kind: "refuted",
            reason: `fold step cannot send (${renderPrimitiveValue(shorter.output)}, ${renderPrimitiveValue(element)}) to both ${renderPrimitiveValue(priorOutput)} and ${renderPrimitiveValue(longer.output)}`,
          };
        }
        continue;
      }

      outputByStep.set(stepKey, longer.output);
      steps.push({
        accumulator: shorter.output,
        element,
        output: longer.output,
      });
    }
  }

  // steps may be empty: fold deduction is weak, and the synthesizer then
  // relies entirely on full-example validation.
  return { kind: "inferred", init, steps };
}

function isPrefix<Element extends PrimitiveValue>(
  prefix: readonly Element[],
  list: readonly Element[],
): boolean {
  return prefix.every((value, index) => {
    const other = list[index];
    return other !== undefined && primitiveValueEquals(value, other);
  });
}
