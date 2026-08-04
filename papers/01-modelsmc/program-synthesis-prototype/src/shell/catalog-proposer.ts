import {
  expressionCost,
  inferType,
  type Expr,
  type Program,
  type StaticType,
} from "../core/language.verify.js";
import type { ProposalContext, ProposalResult, Proposer } from "./proposal.js";
import { renderProgram } from "./render.js";

function intLiteral(value: bigint): Expr {
  return { kind: "IntLiteral", intValue: value };
}

function boolLiteral(value: boolean): Expr {
  return { kind: "BoolLiteral", boolValue: value };
}

function binary(
  kind: "Add" | "Subtract" | "Multiply" | "LessThan" | "EqualInt" | "And",
  left: Expr,
  right: Expr,
): Expr {
  if (kind === "Add") return { kind, left, right };
  if (kind === "Subtract") return { kind, left, right };
  if (kind === "Multiply") return { kind, left, right };
  if (kind === "LessThan") return { kind, left, right };
  if (kind === "EqualInt") return { kind, left, right };
  return { kind, left, right };
}

function addIntegerCandidates(
  candidates: Expr[],
  constants: readonly bigint[],
  variable: Expr = { kind: "Input" },
): void {
  candidates.push(variable);

  // Put affine skeletons early enough that the default deterministic run reaches them.
  // Constants still determine order, so this remains a finite catalog rather than an
  // evaluator-guided oracle.
  for (const intercept of constants) {
    for (const slope of constants) {
      candidates.push(
        binary(
          "Add",
          intLiteral(intercept),
          binary("Multiply", intLiteral(slope), variable),
        ),
      );
    }
  }

  for (const value of constants) {
    const literal = intLiteral(value);
    candidates.push(literal);
    candidates.push(binary("Add", variable, literal));
    candidates.push(binary("Add", literal, variable));
    candidates.push(binary("Subtract", variable, literal));
    candidates.push(binary("Subtract", literal, variable));
    candidates.push(binary("Multiply", literal, variable));
  }

}

function addIntegerPredicateCandidates(
  candidates: Expr[],
  constants: readonly bigint[],
  variable: Expr = { kind: "Input" },
): void {
  for (const value of constants) {
    const literal = intLiteral(value);
    candidates.push(binary("LessThan", variable, literal));
    candidates.push(binary("LessThan", literal, variable));
    candidates.push(binary("EqualInt", variable, literal));
  }
  if (constants.length >= 2) {
    const lower = binary("LessThan", intLiteral(constants[0]!), variable);
    const upper = binary("LessThan", variable, intLiteral(constants[constants.length - 1]!));
    candidates.push(binary("And", lower, upper));
  }
}

function buildScalarCatalog(context: ProposalContext): Expr[] {
  const candidates: Expr[] = [];
  const input: Expr = { kind: "Input" };

  if (context.outputType === "IntType") {
    for (const constant of context.integerConstants) candidates.push(intLiteral(constant));
    if (context.inputType === "IntType") {
      addIntegerCandidates(candidates, context.integerConstants);
      for (const pivot of context.integerConstants) {
        candidates.push({
          kind: "IfThenElse",
          condition: binary("LessThan", input, intLiteral(pivot)),
          thenExpr: intLiteral(context.integerConstants[0]!),
          elseExpr: intLiteral(context.integerConstants[context.integerConstants.length - 1]!),
        });
      }
    } else {
      for (const whenTrue of context.integerConstants) {
        for (const whenFalse of context.integerConstants) {
          candidates.push({
            kind: "IfThenElse",
            condition: input,
            thenExpr: intLiteral(whenTrue),
            elseExpr: intLiteral(whenFalse),
          });
        }
      }
    }
  } else {
    candidates.push(boolLiteral(false), boolLiteral(true));
    if (context.inputType === "BoolType") {
      candidates.push(input, { kind: "Not", operand: input });
      candidates.push(binary("And", input, boolLiteral(false)));
      candidates.push(binary("And", input, boolLiteral(true)));
    } else {
      addIntegerPredicateCandidates(candidates, context.integerConstants);
    }
  }

  return candidates;
}

function mapBodies(
  inputElementType: "IntType" | "BoolType",
  outputElementType: "IntType" | "BoolType",
  constants: readonly bigint[],
): Expr[] {
  const item: Expr = { kind: "Item" };
  const candidates: Expr[] = [];
  if (outputElementType === "IntType") {
    if (inputElementType === "IntType") {
      candidates.push(item);
      // Put the common pointwise affine edits before the wider affine catalog.
      // This does not alter the legacy scalar catalog order.
      for (const constant of constants) {
        candidates.push(binary("Add", item, intLiteral(constant)));
      }
      for (const constant of constants) {
        // Prefer the equivalent addition form when the negated literal is
        // already available (for example, item + 1 over item - -1).
        if (!constants.some((candidate) => candidate === -constant)) {
          candidates.push(binary("Subtract", item, intLiteral(constant)));
        }
      }
      for (const constant of constants) {
        candidates.push(binary("Multiply", item, intLiteral(constant)));
      }
      for (const constant of constants) candidates.push(intLiteral(constant));
      addIntegerCandidates(candidates, constants, item);
    } else {
      for (const constant of constants) candidates.push(intLiteral(constant));
      for (const whenTrue of constants) {
        for (const whenFalse of constants) {
          candidates.push({
            kind: "IfThenElse",
            condition: item,
            thenExpr: intLiteral(whenTrue),
            elseExpr: intLiteral(whenFalse),
          });
        }
      }
    }
  } else {
    candidates.push(boolLiteral(false), boolLiteral(true));
    if (inputElementType === "BoolType") {
      candidates.push(item, { kind: "Not", operand: item });
      candidates.push(binary("And", item, boolLiteral(false)));
      candidates.push(binary("And", item, boolLiteral(true)));
    } else {
      addIntegerPredicateCandidates(candidates, constants, item);
    }
  }
  return candidates;
}

function foldInitials(outputType: StaticType, constants: readonly bigint[]): Expr[] {
  if (outputType === "IntType") return constants.map(intLiteral);
  if (outputType === "BoolType") return [boolLiteral(false), boolLiteral(true)];
  if (outputType === "IntListType") return [{ kind: "EmptyIntList" }];
  return [{ kind: "EmptyBoolList" }];
}

function intFoldReducers(inputElementType: "IntType" | "BoolType", constants: readonly bigint[]): Expr[] {
  const item: Expr = { kind: "Item" };
  const accumulator: Expr = { kind: "Accumulator" };
  const reducers: Expr[] = [accumulator];
  if (inputElementType === "IntType") {
    reducers.push(
      item,
      binary("Add", item, accumulator),
      binary("Add", accumulator, item),
      binary("Subtract", item, accumulator),
      binary("Subtract", accumulator, item),
      binary("Multiply", item, accumulator),
    );
  }
  for (const constant of constants) reducers.push(intLiteral(constant));
  if (inputElementType === "BoolType") {
    for (const whenTrue of constants) {
      for (const whenFalse of constants) {
        reducers.push({
          kind: "IfThenElse",
          condition: item,
          thenExpr: intLiteral(whenTrue),
          elseExpr: accumulator,
        });
        reducers.push({
          kind: "IfThenElse",
          condition: item,
          thenExpr: intLiteral(whenTrue),
          elseExpr: intLiteral(whenFalse),
        });
      }
    }
  }
  return reducers;
}

function boolFoldReducers(inputElementType: "IntType" | "BoolType", constants: readonly bigint[]): Expr[] {
  const item: Expr = { kind: "Item" };
  const accumulator: Expr = { kind: "Accumulator" };
  const reducers: Expr[] = [accumulator, boolLiteral(false), boolLiteral(true)];
  const predicates: Expr[] = [];
  if (inputElementType === "BoolType") {
    predicates.push(item, { kind: "Not", operand: item });
  } else {
    addIntegerPredicateCandidates(predicates, constants, item);
  }
  for (const predicate of predicates) {
    reducers.push(
      predicate,
      binary("And", accumulator, predicate),
      {
        kind: "IfThenElse",
        condition: predicate,
        thenExpr: boolLiteral(true),
        elseExpr: accumulator,
      },
    );
  }
  return reducers;
}

function listFoldReducers(
  inputElementType: "IntType" | "BoolType",
  outputType: "IntListType" | "BoolListType",
  constants: readonly bigint[],
): Expr[] {
  const item: Expr = { kind: "Item" };
  const accumulator: Expr = { kind: "Accumulator" };
  const reducers: Expr[] = [accumulator];
  if (inputElementType === "IntType" && outputType === "IntListType") {
    const prepend: Expr = { kind: "PrependInt", head: item, tail: accumulator };
    reducers.push(prepend);
    for (const pivot of constants) {
      const literal = intLiteral(pivot);
      for (const condition of [
        binary("LessThan", literal, item),
        binary("LessThan", item, literal),
        binary("EqualInt", item, literal),
      ] as const) {
        reducers.push({
          kind: "IfThenElse",
          condition,
          thenExpr: prepend,
          elseExpr: accumulator,
        });
      }
    }
  } else if (inputElementType === "BoolType" && outputType === "BoolListType") {
    const prepend: Expr = { kind: "PrependBool", head: item, tail: accumulator };
    reducers.push(
      prepend,
      { kind: "IfThenElse", condition: item, thenExpr: prepend, elseExpr: accumulator },
      {
        kind: "IfThenElse",
        condition: { kind: "Not", operand: item },
        thenExpr: prepend,
        elseExpr: accumulator,
      },
    );
  } else if (inputElementType === "IntType" && outputType === "BoolListType") {
    for (const pivot of constants) {
      const head = binary("LessThan", item, intLiteral(pivot));
      reducers.push({ kind: "PrependBool", head, tail: accumulator });
    }
  } else {
    for (const whenTrue of constants) {
      const head: Expr = {
        kind: "IfThenElse",
        condition: item,
        thenExpr: intLiteral(whenTrue),
        elseExpr: intLiteral(constants[0]!),
      };
      reducers.push({ kind: "PrependInt", head, tail: accumulator });
    }
  }
  return reducers;
}

function foldReducers(
  inputElementType: "IntType" | "BoolType",
  outputType: StaticType,
  constants: readonly bigint[],
): Expr[] {
  if (outputType === "IntType") return intFoldReducers(inputElementType, constants);
  if (outputType === "BoolType") return boolFoldReducers(inputElementType, constants);
  return listFoldReducers(inputElementType, outputType, constants);
}

function listElementType(type: StaticType): "IntType" | "BoolType" | null {
  if (type === "IntListType") return "IntType";
  if (type === "BoolListType") return "BoolType";
  return null;
}

function buildListCatalog(context: ProposalContext): Program[] {
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

function buildScalarToListCatalog(context: ProposalContext): Program[] {
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

function buildCatalog(context: ProposalContext): Program[] {
  const inputIsList = listElementType(context.inputType) !== null;
  const outputIsList = listElementType(context.outputType) !== null;
  const candidates = inputIsList
    ? buildListCatalog(context)
    : outputIsList
      ? buildScalarToListCatalog(context)
      : buildScalarCatalog(context).map((body): Program => ({
        kind: "ExpressionProgram",
        body,
      }));
  const seen = new Set<string>();
  return candidates.filter((candidate) => {
    const type = inferType(candidate, context.inputType);
    if (type.kind === "TypeError" || type.inferred !== context.outputType) return false;
    if (expressionCost(candidate) > BigInt(context.maxCost)) return false;
    const rendered = renderProgram(candidate, context.inputType);
    if (seen.has(rendered)) return false;
    seen.add(rendered);
    return true;
  });
}

export class CatalogProposer implements Proposer {
  readonly name = "catalog";

  async propose(context: ProposalContext): Promise<ProposalResult> {
    const candidates = buildCatalog(context);
    if (candidates.length === 0) throw new Error("the configured grammar produced no candidates");
    const expression = candidates[context.requestIndex % candidates.length]!;
    return {
      expression,
      rationale: `offline catalog candidate ${context.requestIndex % candidates.length + 1}/${candidates.length}`,
      source: "catalog",
    };
  }
}

export function inferredCatalogSize(
  inputType: StaticType,
  outputType: StaticType,
  constants: readonly bigint[],
  maxCost: number,
): number {
  const placeholderScore = {
    kind: "Scored" as const,
    inferredType: outputType,
    evaluations: [],
    totalLoss: 0,
    exactMatches: 0,
    cost: 1,
    logTarget: 0,
    exactProgram: false,
  };
  return buildCatalog({
    requestIndex: 0,
    inputType,
    outputType,
    examples: [],
    integerConstants: constants,
    maxDepth: 1,
    maxNodes: 1,
    maxCost,
    ancestor: { kind: "ExpressionProgram", body: { kind: "Input" } },
    ancestorScore: placeholderScore,
  }).length;
}
