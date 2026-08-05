import type { Expr } from "../../core/language.verify.js";
import {
  addIntegerCandidates,
  addIntegerPredicateCandidates,
  binary,
  boolLiteral,
  intLiteral,
  type ElementType,
} from "./expressions.js";

export function mapBodies(
  inputElementType: ElementType,
  outputElementType: ElementType,
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
