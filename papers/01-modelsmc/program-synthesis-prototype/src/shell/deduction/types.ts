import type { RuntimeValue } from "../../core/language.verify.js";

export interface DeductionEvent {
  readonly kind:
    | "search.families"
    | "family.viable"
    | "family.refuted"
    | "deduction.inferred"
    | "deduction.partial";
  readonly message: string;
  readonly data: Readonly<Record<string, unknown>>;
}

export interface ExampleLike {
  readonly input: RuntimeValue;
  readonly output: RuntimeValue;
}
export interface DerivedExample {
  readonly inputs: readonly RuntimeValue[];
  readonly output: RuntimeValue;
}
