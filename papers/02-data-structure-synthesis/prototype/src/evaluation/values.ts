import {
  typeEquals,
  type ObjectType,
  type PrimitiveValue,
} from "../ast.js";
import { isClosure } from "./closures.js";
import type { Value } from "./types.js";

export function expectInt(value: Value): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new TypeError("Expected a safe integer value.");
  }
  return value;
}

export function expectBool(value: Value): boolean {
  if (typeof value !== "boolean") {
    throw new TypeError("Expected a boolean value.");
  }
  return value;
}

export function expectString(value: Value): string {
  if (typeof value !== "string") {
    throw new TypeError("Expected a string value.");
  }
  return value;
}

export function expectPrimitive(value: Value): PrimitiveValue {
  if (typeof value === "number") {
    return expectInt(value);
  }
  if (typeof value === "boolean" || typeof value === "string") {
    return value;
  }
  throw new TypeError("Expected a primitive value.");
}

export function expectIntList(value: Value): readonly number[] {
  if (!Array.isArray(value)) {
    throw new TypeError("Expected a list of safe integers.");
  }
  return value.map((element) => expectInt(element));
}

export function valueMatchesType(value: Value, type: ObjectType): boolean {
  switch (type.kind) {
    case "int":
      return typeof value === "number" && Number.isSafeInteger(value);
    case "bool":
      return typeof value === "boolean";
    case "string":
      return typeof value === "string";
    case "list":
      return (
        Array.isArray(value) &&
        value.every((element) => valueMatchesType(element, type.element))
      );
    case "function":
      return (
        isClosure(value) &&
        typeEquals(value.parameterType, type.parameter) &&
        typeEquals(value.resultType, type.result)
      );
  }
}
