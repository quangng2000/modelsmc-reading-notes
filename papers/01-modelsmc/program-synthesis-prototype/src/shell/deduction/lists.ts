import {
  boolValue,
  intValue,
  sameValue,
  type RuntimeValue,
  type StaticType,
} from "../../core/language.verify.js";
import { renderValue } from "../ast/render.js";
import type { DerivedExample } from "./types.js";

export function listElementType(type: StaticType): StaticType | null {
  if (type === "IntListType") return "IntType";
  if (type === "BoolListType") return "BoolType";
  return null;
}

export function listItems(value: RuntimeValue): RuntimeValue[] | null {
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

export function sameList(left: RuntimeValue, right: RuntimeValue): boolean {
  return (
    (left.kind === "IntListValue" || left.kind === "BoolListValue") &&
    (right.kind === "IntListValue" || right.kind === "BoolListValue") &&
    sameValue(left, right)
  );
}

export function formatDerived(examples: readonly DerivedExample[]): string {
  return examples
    .map((example) => {
      const inputs = example.inputs.map(renderValue).join(", ");
      return `${example.inputs.length > 1 ? `(${inputs})` : inputs} -> ${renderValue(example.output)}`;
    })
    .join(", ");
}
