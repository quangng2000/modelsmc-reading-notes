import {
  sameValue,
  type RuntimeValue,
  type StaticType,
} from "../../core/language.verify.js";
import { renderType, renderValue } from "../ast/render.js";
import { formatDerived, listElementType, listItems, sameList } from "./lists.js";
import type { DeductionEvent, DerivedExample, ExampleLike } from "./types.js";

export function deriveFoldEvents(
  inputType: StaticType,
  outputType: StaticType,
  examples: readonly ExampleLike[],
): DeductionEvent[] {
  const inputElement = listElementType(inputType);
  if (inputElement === null) return [];

  const emptyExamples = examples.filter((example) => listItems(example.input)!.length === 0);
  if (emptyExamples.length > 1) {
    const firstOutput = emptyExamples[0]!.output;
    const conflict = emptyExamples.find((example) => !sameValue(firstOutput, example.output));
    if (conflict !== undefined) {
      return [{
        kind: "family.refuted",
        message: `family foldr: refuted by contradictory empty-list outputs ${renderValue(firstOutput)} and ${renderValue(conflict.output)}; ?initial is a single value`,
        data: {
          family: "foldr",
          reason: "initial-contradiction",
          firstOutput: renderValue(firstOutput),
          secondOutput: renderValue(conflict.output),
        },
      }];
    }
  }

  const events: DeductionEvent[] = [{
    kind: "family.viable",
    message: `family foldr: viable skeleton (xs: ${renderType(inputType)}) => foldr((item: ${renderType(inputElement)}, acc: ${renderType(outputType)}) => ?reducer, ?initial, xs); inferred ?reducer: ${renderType(inputElement)} -> ${renderType(outputType)} -> ${renderType(outputType)}; scoped holes exclude outer xs`,
    data: {
      family: "foldr",
      inputElementType: inputElement,
      accumulatorType: outputType,
      skeleton: "FoldRightProgram",
      scopeRestriction: "initial/reducer exclude outer Input",
    },
  }];

  if (emptyExamples.length === 0) {
    events.push({
      kind: "deduction.partial",
      message: "deduction for foldr ?initial: partial; no empty-list example fixes the initial value",
      data: { family: "foldr", hole: "initial", reason: "missing-empty-example" },
    });
  } else {
    events.push({
      kind: "deduction.inferred",
      message: `deduction for foldr ?initial: ${renderValue(emptyExamples[0]!.output)}`,
      data: {
        family: "foldr",
        hole: "initial",
        output: renderValue(emptyExamples[0]!.output),
      },
    });
  }

  const derived: DerivedExample[] = [];
  const reducerMapping: {
    readonly item: RuntimeValue;
    readonly accumulator: RuntimeValue;
    readonly output: RuntimeValue;
  }[] = [];
  let missingSuffixes = 0;
  for (const example of examples) {
    const inputItems = listItems(example.input)!;
    if (inputItems.length === 0) continue;
    if (example.input.kind !== "IntListValue" && example.input.kind !== "BoolListValue") continue;
    const tailInput: RuntimeValue = example.input.kind === "IntListValue"
      ? {
          kind: "IntListValue",
          intListValue:
            example.input.intListValue.kind === "IntCons"
              ? example.input.intListValue.tail
              : example.input.intListValue,
        }
      : {
          kind: "BoolListValue",
          boolListValue:
            example.input.boolListValue.kind === "BoolCons"
              ? example.input.boolListValue.tail
              : example.input.boolListValue,
        };
    const suffixExample = examples.find((candidate) => sameList(candidate.input, tailInput));
    if (suffixExample === undefined) {
      missingSuffixes += 1;
      continue;
    }
    const item = inputItems[0]!;
    const accumulator = suffixExample.output;
    const previous = reducerMapping.find((entry) =>
      sameValue(entry.item, item) && sameValue(entry.accumulator, accumulator),
    );
    if (previous !== undefined && !sameValue(previous.output, example.output)) {
      events.push({
        kind: "family.refuted",
        message: `family foldr: refuted by contradictory reducer examples; (${renderValue(item)}, ${renderValue(accumulator)}) would have to produce both ${renderValue(previous.output)} and ${renderValue(example.output)}`,
        data: {
          family: "foldr",
          reason: "reducer-contradiction",
          item: renderValue(item),
          accumulator: renderValue(accumulator),
          firstOutput: renderValue(previous.output),
          secondOutput: renderValue(example.output),
        },
      });
      return events;
    }
    if (previous === undefined) {
      reducerMapping.push({ item, accumulator, output: example.output });
      derived.push({ inputs: [item, accumulator], output: example.output });
    }
  }

  if (derived.length > 0) {
    events.push({
      kind: "deduction.inferred",
      message: `deduction for foldr ?reducer from observed suffixes: ${formatDerived(derived)}`,
      data: {
        family: "foldr",
        hole: "reducer",
        examples: derived.map((example) => ({
          item: renderValue(example.inputs[0]!),
          accumulator: renderValue(example.inputs[1]!),
          output: renderValue(example.output),
        })),
      },
    });
  }
  if (missingSuffixes > 0 || derived.length === 0) {
    events.push({
      kind: "deduction.partial",
      message: `deduction for foldr ?reducer: partial; ${missingSuffixes} nonempty example${missingSuffixes === 1 ? "" : "s"} lack${missingSuffixes === 1 ? "s" : ""} an observed tail result`,
      data: {
        family: "foldr",
        hole: "reducer",
        reason: "missing-suffix-examples",
        missingSuffixes,
      },
    });
  }
  return events;
}
