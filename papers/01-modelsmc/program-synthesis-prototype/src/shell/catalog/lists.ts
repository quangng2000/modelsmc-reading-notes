import type { Expr, Program } from "../../core/language.verify.js";
import type { ProposalContext } from "../proposal/index.js";
import { boolLiteral, intLiteral, listElementType } from "./expressions.js";
import { foldInitials, foldReducers } from "./fold.js";
import { mapBodies } from "./map.js";

export function buildListCatalog(context: ProposalContext): Program[] {
  const inputElementType = listElementType(context.inputType);
  if (inputElementType === null) return [];
  const candidates: Program[] = [];
  if (context.outputType === "IntType") {
    for (const constant of context.integerConstants) {
      candidates.push({
        kind: "ExpressionProgram",
        body: intLiteral(constant),
      });
    }
  } else if (context.outputType === "BoolType") {
    candidates.push(
      { kind: "ExpressionProgram", body: boolLiteral(false) },
      { kind: "ExpressionProgram", body: boolLiteral(true) },
    );
  } else if (context.outputType === "IntListType") {
    candidates.push({
      kind: "ExpressionProgram",
      body: { kind: "EmptyIntList" },
    });
  } else {
    candidates.push({
      kind: "ExpressionProgram",
      body: { kind: "EmptyBoolList" },
    });
  }
  const outputElementType = listElementType(context.outputType);
  const mapperBodies = outputElementType === null
    ? []
    : mapBodies(inputElementType, outputElementType, context.integerConstants);
  const quickMapperCount = Math.min(
    mapperBodies.length,
    context.integerConstants.length + 3,
  );
  if (outputElementType !== null) {
    candidates.push({ kind: "ExpressionProgram", body: { kind: "Input" } });
    for (const mapper of mapperBodies.slice(0, quickMapperCount)) {
      candidates.push({ kind: "MapProgram", mapper });
    }
  }
  const reducers = foldReducers(inputElementType, context.outputType, context.integerConstants);
  for (const initial of foldInitials(context.outputType, context.integerConstants)) {
    for (const reducer of reducers) {
      candidates.push({ kind: "FoldRightProgram", initial, reducer });
    }
  }
  if (outputElementType !== null) {
    for (const mapper of mapperBodies.slice(quickMapperCount)) {
      candidates.push({ kind: "MapProgram", mapper });
    }
  }
  return candidates;
}

export function buildScalarToListCatalog(context: ProposalContext): Program[] {
  if (context.outputType === "IntListType") {
    const empty: Expr = { kind: "EmptyIntList" };
    const candidates: Program[] = [
      { kind: "ExpressionProgram", body: empty },
      ...context.integerConstants.map((constant): Program => ({
        kind: "ExpressionProgram",
        body: { kind: "PrependInt", head: intLiteral(constant), tail: empty },
      })),
    ];
    if (context.inputType === "IntType") {
      candidates.push({
        kind: "ExpressionProgram",
        body: { kind: "PrependInt", head: { kind: "Input" }, tail: empty },
      });
    }
    return candidates;
  }
  const empty: Expr = { kind: "EmptyBoolList" };
  const candidates: Program[] = [
    { kind: "ExpressionProgram", body: empty },
    {
      kind: "ExpressionProgram",
      body: { kind: "PrependBool", head: boolLiteral(false), tail: empty },
    },
    {
      kind: "ExpressionProgram",
      body: { kind: "PrependBool", head: boolLiteral(true), tail: empty },
    },
  ];
  if (context.inputType === "BoolType") {
    candidates.push({
      kind: "ExpressionProgram",
      body: { kind: "PrependBool", head: { kind: "Input" }, tail: empty },
    });
  }
  return candidates;
}
