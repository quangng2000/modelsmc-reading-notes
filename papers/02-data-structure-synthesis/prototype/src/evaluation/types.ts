import type {
  Expression,
  ObjectType,
  PrimitiveValue,
} from "../ast.js";
import type { TypeBinding } from "../typecheck.js";

export interface ClosureValue {
  readonly kind: "closure";
  readonly parameter: string;
  readonly parameterType: ObjectType;
  readonly resultType: ObjectType;
  readonly body: Expression;
  readonly environment: EvaluationEnvironment;
}

export type Value = PrimitiveValue | readonly Value[] | ClosureValue;

export interface ValueBinding extends TypeBinding {
  readonly value: Value;
}

export type EvaluationEnvironment = readonly ValueBinding[];
