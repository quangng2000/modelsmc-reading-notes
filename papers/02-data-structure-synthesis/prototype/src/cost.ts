import type { BinaryOperator, Expression } from "./ast.js";

export const OPERATOR_COSTS: Readonly<Record<BinaryOperator, number>> = {
  "+": 1,
  "-": 1,
  "*": 2,
};

export function expressionCost(expression: Expression): number {
  switch (expression.kind) {
    case "int":
      return 1;
    case "list":
      return (
        1 +
        expression.elements.reduce(
          (total, element) => total + expressionCost(element),
          0,
        )
      );
    case "variable":
      return 0;
    case "binary":
      return (
        OPERATOR_COSTS[expression.operator] +
        expressionCost(expression.left) +
        expressionCost(expression.right)
      );
    case "lambda":
      return 1 + expressionCost(expression.body);
    case "map":
      return (
        1 +
        expressionCost(expression.mapper) +
        expressionCost(expression.list)
      );
    case "hole":
      return 0;
  }
}
