import type { Expr } from "../../core/language.verify.js";
import type { ProposalContext } from "../proposal/index.js";
import {
  addIntegerCandidates,
  addIntegerPredicateCandidates,
  binary,
  boolLiteral,
  intLiteral,
} from "./expressions.js";

export function buildScalarCatalog(context: ProposalContext): Expr[] {
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
