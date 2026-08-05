import type { StaticType } from "../../core/language.verify.js";
import { renderType } from "../ast/render.js";
import { deriveFoldEvents } from "./fold.js";
import { listElementType } from "./lists.js";
import { deriveMapEvents } from "./map.js";
import type { DeductionEvent, ExampleLike } from "./types.js";

export type { DeductionEvent, ExampleLike } from "./types.js";

export function deriveSynthesisTrace(
  inputType: StaticType,
  outputType: StaticType,
  examples: readonly ExampleLike[],
): readonly DeductionEvent[] {
  const families =
    listElementType(inputType) === null
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
    ...deriveMapEvents(inputType, outputType, examples),
    ...deriveFoldEvents(inputType, outputType, examples),
  ];
}
