import type { Expression } from "../ast.js";

export function sortBucketBySize(bucket: Expression[]): void {
  bucket.sort((left, right) => expressionSize(left) - expressionSize(right));
}

// AST node count, used only as the intra-bucket simplicity tie-break. The
// enumerator itself only produces scalar node kinds, but the switch stays
// exhaustive over the whole Expression union.
function expressionSize(expression: Expression): number {
  switch (expression.kind) {
    case "int":
    case "bool":
    case "string":
    case "variable":
    case "hole":
      return 1;
    case "list":
      return (
        1 +
        expression.elements.reduce(
          (total, element) => total + expressionSize(element),
          0,
        )
      );
    case "binary":
    case "concat":
    case "comparison":
    case "logic":
      return (
        1 + expressionSize(expression.left) + expressionSize(expression.right)
      );
    case "not":
    case "length":
      return 1 + expressionSize(expression.operand);
    case "lambda":
      return 1 + expressionSize(expression.body);
    case "map":
      return (
        1 +
        expressionSize(expression.mapper) +
        expressionSize(expression.list)
      );
    case "filter":
      return (
        1 +
        expressionSize(expression.predicate) +
        expressionSize(expression.list)
      );
    case "fold":
      return (
        1 +
        expressionSize(expression.reducer) +
        expressionSize(expression.initial) +
        expressionSize(expression.list)
      );
  }
}
