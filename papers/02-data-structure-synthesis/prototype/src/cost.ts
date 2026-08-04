import type { ArithmeticOperator, Expression } from "./ast.js";

export const OPERATOR_COSTS: Readonly<Record<ArithmeticOperator, number>> = {
  "+": 1,
  "-": 1,
  "*": 2,
  "%": 2,
};

export function expressionCost(expression: Expression): number {
  switch (expression.kind) {
    case "int":
    case "bool":
    case "string":
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
    case "concat":
      return (
        1 + expressionCost(expression.left) + expressionCost(expression.right)
      );
    case "length":
      return 1 + expressionCost(expression.operand);
    case "comparison":
    case "logic":
      return (
        1 + expressionCost(expression.left) + expressionCost(expression.right)
      );
    case "not":
      return 1 + expressionCost(expression.operand);
    case "lambda":
      return 1 + expressionCost(expression.body);
    case "map":
      return (
        1 +
        expressionCost(expression.mapper) +
        expressionCost(expression.list)
      );
    case "filter":
      return (
        1 +
        expressionCost(expression.predicate) +
        expressionCost(expression.list)
      );
    case "fold":
      return (
        1 +
        expressionCost(expression.reducer) +
        expressionCost(expression.initial) +
        expressionCost(expression.list)
      );
    case "hole":
      return 0;
  }
}
