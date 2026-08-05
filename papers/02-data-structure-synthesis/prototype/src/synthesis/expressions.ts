import { typeEquals, type Expression } from "../ast.js";
import { inferType, type TypeEnvironment } from "../typecheck.js";

export function substituteHole(
  expression: Expression,
  holeName: string,
  replacement: Expression,
): Expression {
  return substituteHoleInContext(expression, holeName, replacement, []);
}

function substituteHoleInContext(
  expression: Expression,
  holeName: string,
  replacement: Expression,
  environment: TypeEnvironment,
): Expression {
  switch (expression.kind) {
    case "int":
    case "bool":
    case "string":
    case "variable":
      return expression;
    case "hole": {
      if (expression.name !== holeName) {
        return expression;
      }

      const replacementType = inferType(replacement, environment);
      if (
        replacementType === undefined ||
        !typeEquals(replacementType, expression.expectedType)
      ) {
        throw new TypeError(`Replacement for ?${holeName} has the wrong type.`);
      }
      return replacement;
    }
    case "list":
      return {
        ...expression,
        elements: expression.elements.map((element) =>
          substituteHoleInContext(
            element,
            holeName,
            replacement,
            environment,
          ),
        ),
      };
    case "binary":
    case "concat":
    case "comparison":
    case "logic":
      return {
        ...expression,
        left: substituteHoleInContext(
          expression.left,
          holeName,
          replacement,
          environment,
        ),
        right: substituteHoleInContext(
          expression.right,
          holeName,
          replacement,
          environment,
        ),
      };
    case "not":
      return {
        ...expression,
        operand: substituteHoleInContext(
          expression.operand,
          holeName,
          replacement,
          environment,
        ),
      };
    case "length":
      return {
        ...expression,
        operand: substituteHoleInContext(
          expression.operand,
          holeName,
          replacement,
          environment,
        ),
      };
    case "lambda":
      return {
        ...expression,
        body: substituteHoleInContext(
          expression.body,
          holeName,
          replacement,
          [
            ...environment,
            {
              name: expression.parameter,
              type: expression.parameterType,
            },
          ],
        ),
      };
    case "map":
      return {
        ...expression,
        mapper: substituteHoleInContext(
          expression.mapper,
          holeName,
          replacement,
          environment,
        ),
        list: substituteHoleInContext(
          expression.list,
          holeName,
          replacement,
          environment,
        ),
      };
    case "filter":
      return {
        ...expression,
        predicate: substituteHoleInContext(
          expression.predicate,
          holeName,
          replacement,
          environment,
        ),
        list: substituteHoleInContext(
          expression.list,
          holeName,
          replacement,
          environment,
        ),
      };
    case "fold":
      return {
        ...expression,
        reducer: substituteHoleInContext(
          expression.reducer,
          holeName,
          replacement,
          environment,
        ),
        initial: substituteHoleInContext(
          expression.initial,
          holeName,
          replacement,
          environment,
        ),
        list: substituteHoleInContext(
          expression.list,
          holeName,
          replacement,
          environment,
        ),
      };
  }
}

export function substituteVariable(
  expression: Expression,
  variableName: string,
  replacement: Expression,
): Expression {
  switch (expression.kind) {
    case "int":
    case "bool":
    case "string":
    case "hole":
      return expression;
    case "variable":
      return expression.name === variableName ? replacement : expression;
    case "list":
      return {
        ...expression,
        elements: expression.elements.map((element) =>
          substituteVariable(element, variableName, replacement),
        ),
      };
    case "binary":
    case "concat":
    case "comparison":
    case "logic":
      return {
        ...expression,
        left: substituteVariable(expression.left, variableName, replacement),
        right: substituteVariable(expression.right, variableName, replacement),
      };
    case "not":
      return {
        ...expression,
        operand: substituteVariable(
          expression.operand,
          variableName,
          replacement,
        ),
      };
    case "length":
      return {
        ...expression,
        operand: substituteVariable(
          expression.operand,
          variableName,
          replacement,
        ),
      };
    case "lambda":
      return expression.parameter === variableName
        ? expression
        : {
            ...expression,
            body: substituteVariable(
              expression.body,
              variableName,
              replacement,
            ),
          };
    case "map":
      return {
        ...expression,
        mapper: substituteVariable(
          expression.mapper,
          variableName,
          replacement,
        ),
        list: substituteVariable(expression.list, variableName, replacement),
      };
    case "filter":
      return {
        ...expression,
        predicate: substituteVariable(
          expression.predicate,
          variableName,
          replacement,
        ),
        list: substituteVariable(expression.list, variableName, replacement),
      };
    case "fold":
      return {
        ...expression,
        reducer: substituteVariable(
          expression.reducer,
          variableName,
          replacement,
        ),
        initial: substituteVariable(
          expression.initial,
          variableName,
          replacement,
        ),
        list: substituteVariable(expression.list, variableName, replacement),
      };
  }
}
