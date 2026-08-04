import {
  INT,
  primitiveValueEquals,
  renderPrimitiveValue,
  type PrimitiveType,
  type PrimitiveValue,
} from "../ast.js";
import { validatePrimitiveList, type DeductionResult } from "./types.js";

export interface FilterExample<Element extends PrimitiveValue = number> {
  readonly input: readonly Element[];
  readonly output: readonly Element[];
}

export interface PredicateExample<Element extends PrimitiveValue = number> {
  readonly input: Element;
  readonly output: boolean;
}

export type FilterDeductionResult<Element extends PrimitiveValue = number> =
  DeductionResult<PredicateExample<Element>>;

export function deduceFilterExamples<Element extends PrimitiveValue = number>(
  examples: readonly FilterExample<Element>[],
  elementType: PrimitiveType = INT,
): FilterDeductionResult<Element> {
  const keptByValue = new Map<Element, boolean>();
  const inferred: PredicateExample<Element>[] = [];

  for (const example of examples) {
    validatePrimitiveList(example.input, elementType, "filter input");
    validatePrimitiveList(example.output, elementType, "filter output");

    const inputCounts = countValues(example.input);
    const outputCounts = countValues(example.output);

    for (const [value, outputCount] of outputCounts) {
      if ((inputCounts.get(value) ?? 0) < outputCount) {
        return {
          kind: "refuted",
          reason: `filter cannot introduce ${renderPrimitiveValue(value)}; the output needs more copies than the input provides`,
        };
      }
    }

    for (const [value, inputCount] of inputCounts) {
      const outputCount = outputCounts.get(value) ?? 0;

      // A pure predicate on element values keeps every copy of a value or none.
      if (outputCount !== 0 && outputCount !== inputCount) {
        return {
          kind: "refuted",
          reason: `a pure predicate must keep all or none of the ${inputCount} copies of ${renderPrimitiveValue(value)}, not ${outputCount}`,
        };
      }

      const kept = outputCount === inputCount;
      const priorKept = keptByValue.get(value);
      if (priorKept !== undefined && priorKept !== kept) {
        return {
          kind: "refuted",
          reason: `filter cannot both keep and drop ${renderPrimitiveValue(value)} across examples`,
        };
      }

      if (priorKept === undefined) {
        keptByValue.set(value, kept);
        inferred.push({ input: value, output: kept });
      }
    }

    // Counts alone permit reorderings, so check the surviving order exactly.
    const reconstructed = example.input.filter(
      (value) => keptByValue.get(value) === true,
    );
    if (!listsEqual(reconstructed, example.output)) {
      return {
        kind: "refuted",
        reason: "filter preserves element order",
      };
    }
  }

  return { kind: "inferred", examples: inferred };
}

function countValues<Element extends PrimitiveValue>(
  values: readonly Element[],
): Map<Element, number> {
  const counts = new Map<Element, number>();
  for (const value of values) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return counts;
}

function listsEqual<Element extends PrimitiveValue>(
  left: readonly Element[],
  right: readonly Element[],
): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => {
      const other = right[index];
      return other !== undefined && primitiveValueEquals(value, other);
    })
  );
}
