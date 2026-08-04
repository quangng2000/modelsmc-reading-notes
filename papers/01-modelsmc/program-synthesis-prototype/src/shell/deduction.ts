import {
  boolValue,
  intValue,
  sameValue,
  type RuntimeValue,
  type StaticType,
} from "../core/language.verify.js";
import { renderType, renderValue } from "./render.js";

export interface DeductionEvent {
  readonly kind:
    | "search.families"
    | "family.viable"
    | "family.refuted"
    | "deduction.inferred"
    | "deduction.partial";
  readonly message: string;
  readonly data: Readonly<Record<string, unknown>>;
}

interface ExampleLike {
  readonly input: RuntimeValue;
  readonly output: RuntimeValue;
}

interface DerivedExample {
  readonly inputs: readonly RuntimeValue[];
  readonly output: RuntimeValue;
}

function listElementType(type: StaticType): StaticType | null {
  if (type === "IntListType") return "IntType";
  if (type === "BoolListType") return "BoolType";
  return null;
}

function listItems(value: RuntimeValue): RuntimeValue[] | null {
  if (value.kind === "IntListValue") {
    const items: RuntimeValue[] = [];
    let remaining = value.intListValue;
    while (remaining.kind === "IntCons") {
      items.push(intValue(remaining.head));
      remaining = remaining.tail;
    }
    return items;
  }
  if (value.kind === "BoolListValue") {
    const items: RuntimeValue[] = [];
    let remaining = value.boolListValue;
    while (remaining.kind === "BoolCons") {
      items.push(boolValue(remaining.head));
      remaining = remaining.tail;
    }
    return items;
  }
  return null;
}

function sameList(left: RuntimeValue, right: RuntimeValue): boolean {
  return (
    (left.kind === "IntListValue" || left.kind === "BoolListValue") &&
    (right.kind === "IntListValue" || right.kind === "BoolListValue") &&
    sameValue(left, right)
  );
}

function formatDerived(examples: readonly DerivedExample[]): string {
  return examples
    .map((example) => {
      const inputs = example.inputs.map(renderValue).join(", ");
      return `${example.inputs.length > 1 ? `(${inputs})` : inputs} -> ${renderValue(example.output)}`;
    })
    .join(", ");
}

function mapEvents(
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

function foldEvents(
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
    if (
      example.input.kind !== "IntListValue" &&
      example.input.kind !== "BoolListValue"
    ) {
      continue;
    }
    const tailInput: RuntimeValue = example.input.kind === "IntListValue"
      ? { kind: "IntListValue", intListValue: example.input.intListValue.kind === "IntCons"
        ? example.input.intListValue.tail
        : example.input.intListValue }
      : { kind: "BoolListValue", boolListValue: example.input.boolListValue.kind === "BoolCons"
        ? example.input.boolListValue.tail
        : example.input.boolListValue };
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

export function deriveSynthesisTrace(
  inputType: StaticType,
  outputType: StaticType,
  examples: readonly ExampleLike[],
): readonly DeductionEvent[] {
  const families = listElementType(inputType) === null
    ? ["expression"]
    : listElementType(outputType) === null
      ? ["expression", "foldr"]
      : ["expression", "map", "foldr"];
  return [
    {
      kind: "search.families",
      message: `search started; possible families: ${families.join(", ")}`,
      data: { families, inputType, outputType },
    },
    {
      kind: "family.viable",
      message: `family expression: viable skeleton (${listElementType(inputType) === null ? "x" : "xs"}: ${renderType(inputType)}) => ?body; required ?body: ${renderType(outputType)}`,
      data: {
        family: "expression",
        inputType,
        outputType,
        skeleton: "ExpressionProgram",
      },
    },
    ...mapEvents(inputType, outputType, examples),
    ...foldEvents(inputType, outputType, examples),
  ];
}
