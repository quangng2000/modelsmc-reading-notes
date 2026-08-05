import type { Expression } from "../ast.js";

export function containsHole(expression: Expression): boolean {
  switch (expression.kind) {
    case "int":
    case "bool":
    case "string":
    case "variable":
      return false;
    case "hole":
      return true;
    case "list":
      return expression.elements.some(containsHole);
    case "binary":
    case "concat":
    case "comparison":
    case "logic":
      return containsHole(expression.left) || containsHole(expression.right);
    case "not":
      return containsHole(expression.operand);
    case "length":
      return containsHole(expression.operand);
    case "lambda":
      return containsHole(expression.body);
    case "map":
      return containsHole(expression.mapper) || containsHole(expression.list);
    case "filter":
      return (
        containsHole(expression.predicate) || containsHole(expression.list)
      );
    case "fold":
      return (
        containsHole(expression.reducer) ||
        containsHole(expression.initial) ||
        containsHole(expression.list)
      );
  }
}
