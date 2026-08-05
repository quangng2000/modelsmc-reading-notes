import {
  sameValue,
  type RuntimeValue,
  type StaticType,
} from "../../core/language.verify.js";
import { renderType, renderValue } from "../ast/render.js";
import { formatDerived, listElementType, listItems } from "./lists.js";
import type { DeductionEvent, DerivedExample, ExampleLike } from "./types.js";

export function deriveMapEvents(
  inputType: StaticType,
  outputType: StaticType,
  examples: readonly ExampleLike[],
): DeductionEvent[] {
  const inputElement = listElementType(inputType);
  const outputElement = listElementType(outputType);
  if (inputElement === null || outputElement === null) return [];

  const events: DeductionEvent[] = [];
  const derived: DerivedExample[] = [];
  const mapping: { readonly input: RuntimeValue; readonly output: RuntimeValue }[] = [];
  for (let index = 0; index < examples.length; index += 1) {
    const inputItems = listItems(examples[index]!.input)!;
    const outputItems = listItems(examples[index]!.output)!;
    if (inputItems.length !== outputItems.length) {
      events.push({
        kind: "family.refuted",
        message: `family map: refuted by example ${index + 1}; map preserves length, but ${inputItems.length} input items would need to produce ${outputItems.length} output items`,
        data: {
          family: "map",
          reason: "length-mismatch",
          example: index + 1,
          inputLength: inputItems.length,
          outputLength: outputItems.length,
        },
      });
      return events;
    }
    for (let itemIndex = 0; itemIndex < inputItems.length; itemIndex += 1) {
      const item = inputItems[itemIndex]!;
      const output = outputItems[itemIndex]!;
      const previous = mapping.find((entry) => sameValue(entry.input, item));
      if (previous !== undefined && !sameValue(previous.output, output)) {
        events.push({
          kind: "family.refuted",
          message: `family map: refuted by contradictory mapper examples; ${renderValue(item)} would have to map to both ${renderValue(previous.output)} and ${renderValue(output)}`,
          data: {
            family: "map",
            reason: "duplicate-input-contradiction",
            input: renderValue(item),
            firstOutput: renderValue(previous.output),
            secondOutput: renderValue(output),
          },
        });
        return events;
      }
      if (previous === undefined) {
        mapping.push({ input: item, output });
        derived.push({ inputs: [item], output });
      }
    }
  }

  events.push({
    kind: "family.viable",
    message: `family map: viable skeleton (xs: ${renderType(inputType)}) => map((item: ${renderType(inputElement)}) => ?mapper, xs); inferred ?mapper: ${renderType(inputElement)} -> ${renderType(outputElement)}`,
    data: {
      family: "map",
      inputElementType: inputElement,
      outputElementType: outputElement,
      skeleton: "MapProgram",
    },
  });
  events.push({
    kind: "deduction.inferred",
    message:
      derived.length === 0
        ? "deduction for map ?mapper: no element examples (all observed lists are empty)"
        : `deduction for map ?mapper: ${formatDerived(derived)}`,
    data: {
      family: "map",
      hole: "mapper",
      examples: derived.map((example) => ({
        input: renderValue(example.inputs[0]!),
        output: renderValue(example.output),
      })),
    },
  });
  return events;
}
