import type { Expr, StaticType } from "../../core/language.verify.js";

export type ElementType = "IntType" | "BoolType";

export function intLiteral(value: bigint): Expr {
  return { kind: "IntLiteral", intValue: value };
}

export function boolLiteral(value: boolean): Expr {
  return { kind: "BoolLiteral", boolValue: value };
}

export function binary(
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

export function addIntegerCandidates(
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

export function addIntegerPredicateCandidates(
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

export function listElementType(type: StaticType): ElementType | null {
  if (type === "IntListType") return "IntType";
  if (type === "BoolListType") return "BoolType";
  return null;
}
