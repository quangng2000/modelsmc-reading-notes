import type {
  ArithmeticOperator,
  ComparisonOperator,
  Expression,
  LogicOperator,
} from "../ast.js";
import type { PrimitiveTypeName } from "./types.js";

export const VARIABLE: Expression = { kind: "variable", name: "x" };
export const DEFAULT_CONSTANTS: readonly number[] = [-1, 0, 1, 2];
export const DEFAULT_STRING_CONSTANTS: readonly string[] = [""];
export const DEFAULT_MAX_COST = 4;

export const DEFAULT_VARIABLES: readonly string[] = ["x"];
export const DEFAULT_TARGET_TYPE: PrimitiveTypeName = "int";
export const ARITHMETIC_OPERATORS: readonly ArithmeticOperator[] = [
  "+",
  "-",
  "*",
  "%",
];
export const COMPARISON_OPERATORS: readonly ComparisonOperator[] = [
  "<",
  "<=",
  "==",
];
export const LOGIC_OPERATORS: readonly LogicOperator[] = ["&&", "||"];
export const CONSTANT_COST = 1;
export const BOOL_LITERAL_COST = 1;
export const COMPARISON_COST = 1;
export const LOGIC_COST = 1;
export const NOT_COST = 1;
export const CONCAT_COST = 1;
export const LENGTH_COST = 1;
