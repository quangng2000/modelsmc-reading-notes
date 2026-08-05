//@ backend dafny

export type StaticType =
  | "IntType"
  | "BoolType"
  | "IntListType"
  | "BoolListType";

export type IntList =
  | { kind: "IntNil" }
  | { kind: "IntCons"; head: bigint; tail: IntList };

export type BoolList =
  | { kind: "BoolNil" }
  | { kind: "BoolCons"; head: boolean; tail: BoolList };

export type RuntimeValue =
  | { kind: "IntValue"; intValue: bigint }
  | { kind: "BoolValue"; boolValue: boolean }
  | { kind: "IntListValue"; intListValue: IntList }
  | { kind: "BoolListValue"; boolListValue: BoolList };

export type BindingType =
  | { kind: "UnboundType" }
  | { kind: "BoundType"; bindingType: StaticType };

export type BindingValue =
  | { kind: "UnboundValue" }
  | { kind: "BoundValue"; bindingValue: RuntimeValue };

/**
 * A first-order expression body. Item and Accumulator are scoped variables:
 * Item is bound by map/foldr and Accumulator is bound only by foldr.
 */
export type Expr =
  | { kind: "Input" }
  | { kind: "Item" }
  | { kind: "Accumulator" }
  | { kind: "IntLiteral"; intValue: bigint }
  | { kind: "BoolLiteral"; boolValue: boolean }
  | { kind: "EmptyIntList" }
  | { kind: "EmptyBoolList" }
  | { kind: "PrependInt"; head: Expr; tail: Expr }
  | { kind: "PrependBool"; head: Expr; tail: Expr }
  | { kind: "Add"; left: Expr; right: Expr }
  | { kind: "Subtract"; left: Expr; right: Expr }
  | { kind: "Multiply"; left: Expr; right: Expr }
  | { kind: "LessThan"; left: Expr; right: Expr }
  | { kind: "EqualInt"; left: Expr; right: Expr }
  | { kind: "Not"; operand: Expr }
  | { kind: "And"; left: Expr; right: Expr }
  | { kind: "IfThenElse"; condition: Expr; thenExpr: Expr; elseExpr: Expr };

/**
 * Complete synthesis programs. A plain Expr is the original scalar/list
 * first-order program. Map and foldr carry scoped bodies but are not general
 * first-class Lambda/Apply nodes.
 */
export type Program =
  | { kind: "ExpressionProgram"; body: Expr }
  | { kind: "MapProgram"; mapper: Expr }
  | { kind: "FoldRightProgram"; initial: Expr; reducer: Expr };

export type TypeResult =
  | { kind: "TypeOk"; inferred: StaticType }
  | { kind: "TypeError" };

export type EvalResult =
  | { kind: "EvalOk"; output: RuntimeValue }
  | { kind: "EvalError" };

export interface Example {
  input: RuntimeValue;
  output: RuntimeValue;
}

//@ pure
export function typeOk(inferred: StaticType): TypeResult {
  return { kind: "TypeOk", inferred };
}

//@ pure
export function typeError(): TypeResult {
  return { kind: "TypeError" };
}

//@ pure
export function unboundType(): BindingType {
  return { kind: "UnboundType" };
}

//@ pure
export function boundType(bindingType: StaticType): BindingType {
  return { kind: "BoundType", bindingType };
}

//@ pure
export function unboundValue(): BindingValue {
  return { kind: "UnboundValue" };
}

//@ pure
export function boundValue(bindingValue: RuntimeValue): BindingValue {
  return { kind: "BoundValue", bindingValue };
}

//@ pure
export function intValue(intValue: bigint): RuntimeValue {
  return { kind: "IntValue", intValue };
}

//@ pure
export function boolValue(boolValue: boolean): RuntimeValue {
  return { kind: "BoolValue", boolValue };
}

//@ pure
export function intNil(): IntList {
  return { kind: "IntNil" };
}

//@ pure
export function intCons(head: bigint, tail: IntList): IntList {
  return { kind: "IntCons", head, tail };
}

//@ pure
export function boolNil(): BoolList {
  return { kind: "BoolNil" };
}

//@ pure
export function boolCons(head: boolean, tail: BoolList): BoolList {
  return { kind: "BoolCons", head, tail };
}

//@ pure
export function intListValue(listValue: IntList): RuntimeValue {
  return { kind: "IntListValue", intListValue: listValue };
}

//@ pure
export function boolListValue(listValue: BoolList): RuntimeValue {
  return { kind: "BoolListValue", boolListValue: listValue };
}

//@ pure
export function evalOk(output: RuntimeValue): EvalResult {
  return { kind: "EvalOk", output };
}

//@ pure
export function evalError(): EvalResult {
  return { kind: "EvalError" };
}

//@ pure
export function valueType(value: RuntimeValue): StaticType {
  if (value.kind === "IntValue") return "IntType";
  if (value.kind === "BoolValue") return "BoolType";
  if (value.kind === "IntListValue") return "IntListType";
  return "BoolListType";
}

//@ pure
export function valueMatchesType(value: RuntimeValue, expected: StaticType): boolean {
  if (expected === "IntType") return value.kind === "IntValue";
  if (expected === "BoolType") return value.kind === "BoolValue";
  if (expected === "IntListType") return value.kind === "IntListValue";
  return value.kind === "BoolListValue";
}

//@ pure
export function bindingMatchesType(value: BindingValue, expected: BindingType): boolean {
  if (expected.kind === "UnboundType") return value.kind === "UnboundValue";
  if (value.kind === "UnboundValue") return false;
  return valueMatchesType(value.bindingValue, expected.bindingType);
}

//@ pure
export function sameIntList(left: IntList, right: IntList): boolean {
  if (left.kind === "IntNil") return right.kind === "IntNil";
  return (
    right.kind === "IntCons" &&
    left.head === right.head &&
    sameIntList(left.tail, right.tail)
  );
}

//@ pure
export function sameBoolList(left: BoolList, right: BoolList): boolean {
  if (left.kind === "BoolNil") return right.kind === "BoolNil";
  return (
    right.kind === "BoolCons" &&
    left.head === right.head &&
    sameBoolList(left.tail, right.tail)
  );
}

//@ pure
export function sameValue(left: RuntimeValue, right: RuntimeValue): boolean {
  if (left.kind === "IntValue") {
    return right.kind === "IntValue" && left.intValue === right.intValue;
  }
  if (left.kind === "BoolValue") {
    return right.kind === "BoolValue" && left.boolValue === right.boolValue;
  }
  if (left.kind === "IntListValue") {
    return (
      right.kind === "IntListValue" &&
      sameIntList(left.intListValue, right.intListValue)
    );
  }
  return (
    right.kind === "BoolListValue" &&
    sameBoolList(left.boolListValue, right.boolListValue)
  );
}

//@ pure
export function inferExpression(
  expr: Expr,
  inputType: StaticType,
  itemType: BindingType,
  accumulatorType: BindingType,
): TypeResult {
  if (expr.kind === "Input") return typeOk(inputType);
  if (expr.kind === "Item") {
    if (itemType.kind === "BoundType") return typeOk(itemType.bindingType);
    return typeError();
  }
  if (expr.kind === "Accumulator") {
    if (accumulatorType.kind === "BoundType") {
      return typeOk(accumulatorType.bindingType);
    }
    return typeError();
  }
  if (expr.kind === "IntLiteral") return typeOk("IntType");
  if (expr.kind === "BoolLiteral") return typeOk("BoolType");
  if (expr.kind === "EmptyIntList") return typeOk("IntListType");
  if (expr.kind === "EmptyBoolList") return typeOk("BoolListType");

  if (
    expr.kind === "Add" ||
    expr.kind === "Subtract" ||
    expr.kind === "Multiply" ||
    expr.kind === "LessThan" ||
    expr.kind === "EqualInt"
  ) {
    const left = inferExpression(expr.left, inputType, itemType, accumulatorType);
    const right = inferExpression(expr.right, inputType, itemType, accumulatorType);
    if (
      left.kind === "TypeOk" &&
      left.inferred === "IntType" &&
      right.kind === "TypeOk" &&
      right.inferred === "IntType"
    ) {
      if (
        expr.kind === "Add" ||
        expr.kind === "Subtract" ||
        expr.kind === "Multiply"
      ) {
        return typeOk("IntType");
      }
      return typeOk("BoolType");
    }
    return typeError();
  }

  if (expr.kind === "Not") {
    const operand = inferExpression(expr.operand, inputType, itemType, accumulatorType);
    if (operand.kind === "TypeOk" && operand.inferred === "BoolType") {
      return typeOk("BoolType");
    }
    return typeError();
  }

  if (expr.kind === "And") {
    const left = inferExpression(expr.left, inputType, itemType, accumulatorType);
    const right = inferExpression(expr.right, inputType, itemType, accumulatorType);
    if (
      left.kind === "TypeOk" &&
      left.inferred === "BoolType" &&
      right.kind === "TypeOk" &&
      right.inferred === "BoolType"
    ) {
      return typeOk("BoolType");
    }
    return typeError();
  }

  if (expr.kind === "PrependInt") {
    const head = inferExpression(expr.head, inputType, itemType, accumulatorType);
    const tail = inferExpression(expr.tail, inputType, itemType, accumulatorType);
    if (
      head.kind === "TypeOk" &&
      head.inferred === "IntType" &&
      tail.kind === "TypeOk" &&
      tail.inferred === "IntListType"
    ) {
      return typeOk("IntListType");
    }
    return typeError();
  }

  if (expr.kind === "PrependBool") {
    const head = inferExpression(expr.head, inputType, itemType, accumulatorType);
    const tail = inferExpression(expr.tail, inputType, itemType, accumulatorType);
    if (
      head.kind === "TypeOk" &&
      head.inferred === "BoolType" &&
      tail.kind === "TypeOk" &&
      tail.inferred === "BoolListType"
    ) {
      return typeOk("BoolListType");
    }
    return typeError();
  }

  const condition = inferExpression(expr.condition, inputType, itemType, accumulatorType);
  const thenType = inferExpression(expr.thenExpr, inputType, itemType, accumulatorType);
  const elseType = inferExpression(expr.elseExpr, inputType, itemType, accumulatorType);
  if (
    condition.kind === "TypeOk" &&
    condition.inferred === "BoolType" &&
    thenType.kind === "TypeOk" &&
    elseType.kind === "TypeOk" &&
    thenType.inferred === elseType.inferred
  ) {
    return typeOk(thenType.inferred);
  }
  return typeError();
}

//@ pure
export function evaluateExpression(
  expr: Expr,
  input: RuntimeValue,
  item: BindingValue,
  accumulator: BindingValue,
): EvalResult {
  if (expr.kind === "Input") return evalOk(input);
  if (expr.kind === "Item") {
    if (item.kind === "BoundValue") return evalOk(item.bindingValue);
    return evalError();
  }
  if (expr.kind === "Accumulator") {
    if (accumulator.kind === "BoundValue") return evalOk(accumulator.bindingValue);
    return evalError();
  }
  if (expr.kind === "IntLiteral") return evalOk(intValue(expr.intValue));
  if (expr.kind === "BoolLiteral") return evalOk(boolValue(expr.boolValue));
  if (expr.kind === "EmptyIntList") return evalOk(intListValue(intNil()));
  if (expr.kind === "EmptyBoolList") return evalOk(boolListValue(boolNil()));

  if (
    expr.kind === "Add" ||
    expr.kind === "Subtract" ||
    expr.kind === "Multiply" ||
    expr.kind === "LessThan" ||
    expr.kind === "EqualInt"
  ) {
    const left = evaluateExpression(expr.left, input, item, accumulator);
    const right = evaluateExpression(expr.right, input, item, accumulator);
    if (
      left.kind === "EvalOk" &&
      left.output.kind === "IntValue" &&
      right.kind === "EvalOk" &&
      right.output.kind === "IntValue"
    ) {
      if (expr.kind === "Add") {
        return evalOk(intValue(left.output.intValue + right.output.intValue));
      }
      if (expr.kind === "Subtract") {
        return evalOk(intValue(left.output.intValue - right.output.intValue));
      }
      if (expr.kind === "Multiply") {
        return evalOk(intValue(left.output.intValue * right.output.intValue));
      }
      if (expr.kind === "LessThan") {
        return evalOk(boolValue(left.output.intValue < right.output.intValue));
      }
      return evalOk(boolValue(left.output.intValue === right.output.intValue));
    }
    return evalError();
  }

  if (expr.kind === "Not") {
    const operand = evaluateExpression(expr.operand, input, item, accumulator);
    if (operand.kind === "EvalOk" && operand.output.kind === "BoolValue") {
      return evalOk(boolValue(!operand.output.boolValue));
    }
    return evalError();
  }

  if (expr.kind === "And") {
    const left = evaluateExpression(expr.left, input, item, accumulator);
    const right = evaluateExpression(expr.right, input, item, accumulator);
    if (
      left.kind === "EvalOk" &&
      left.output.kind === "BoolValue" &&
      right.kind === "EvalOk" &&
      right.output.kind === "BoolValue"
    ) {
      return evalOk(boolValue(left.output.boolValue && right.output.boolValue));
    }
    return evalError();
  }

  if (expr.kind === "PrependInt") {
    const head = evaluateExpression(expr.head, input, item, accumulator);
    const tail = evaluateExpression(expr.tail, input, item, accumulator);
    if (
      head.kind === "EvalOk" &&
      head.output.kind === "IntValue" &&
      tail.kind === "EvalOk" &&
      tail.output.kind === "IntListValue"
    ) {
      return evalOk(
        intListValue(intCons(head.output.intValue, tail.output.intListValue)),
      );
    }
    return evalError();
  }

  if (expr.kind === "PrependBool") {
    const head = evaluateExpression(expr.head, input, item, accumulator);
    const tail = evaluateExpression(expr.tail, input, item, accumulator);
    if (
      head.kind === "EvalOk" &&
      head.output.kind === "BoolValue" &&
      tail.kind === "EvalOk" &&
      tail.output.kind === "BoolListValue"
    ) {
      return evalOk(
        boolListValue(boolCons(head.output.boolValue, tail.output.boolListValue)),
      );
    }
    return evalError();
  }

  const condition = evaluateExpression(expr.condition, input, item, accumulator);
  if (condition.kind !== "EvalOk" || condition.output.kind !== "BoolValue") {
    return evalError();
  }
  if (condition.output.boolValue) {
    return evaluateExpression(expr.thenExpr, input, item, accumulator);
  }
  return evaluateExpression(expr.elseExpr, input, item, accumulator);
}

//@ pure
export function listItemType(inputType: StaticType): TypeResult {
  if (inputType === "IntListType") return typeOk("IntType");
  if (inputType === "BoolListType") return typeOk("BoolType");
  return typeError();
}

//@ pure
export function listTypeForScalar(scalarType: StaticType): TypeResult {
  if (scalarType === "IntType") return typeOk("IntListType");
  if (scalarType === "BoolType") return typeOk("BoolListType");
  return typeError();
}

//@ pure
export function expressionUsesInput(expr: Expr): boolean {
  if (expr.kind === "Input") return true;
  if (
    expr.kind === "Item" ||
    expr.kind === "Accumulator" ||
    expr.kind === "IntLiteral" ||
    expr.kind === "BoolLiteral" ||
    expr.kind === "EmptyIntList" ||
    expr.kind === "EmptyBoolList"
  ) {
    return false;
  }
  if (expr.kind === "Not") return expressionUsesInput(expr.operand);
  if (expr.kind === "IfThenElse") {
    return (
      expressionUsesInput(expr.condition) ||
      expressionUsesInput(expr.thenExpr) ||
      expressionUsesInput(expr.elseExpr)
    );
  }
  if (expr.kind === "PrependInt" || expr.kind === "PrependBool") {
    return expressionUsesInput(expr.head) || expressionUsesInput(expr.tail);
  }
  return expressionUsesInput(expr.left) || expressionUsesInput(expr.right);
}

//@ pure
export function inferType(program: Program, inputType: StaticType): TypeResult {
  if (program.kind === "MapProgram") {
    if (expressionUsesInput(program.mapper)) return typeError();
    const itemType = listItemType(inputType);
    if (itemType.kind === "TypeError") return typeError();
    const mapperType = inferExpression(
      program.mapper,
      inputType,
      boundType(itemType.inferred),
      unboundType(),
    );
    if (mapperType.kind === "TypeError") return typeError();
    return listTypeForScalar(mapperType.inferred);
  }

  if (program.kind === "FoldRightProgram") {
    if (
      expressionUsesInput(program.initial) ||
      expressionUsesInput(program.reducer)
    ) {
      return typeError();
    }
    const itemType = listItemType(inputType);
    if (itemType.kind === "TypeError") return typeError();
    const initialType = inferExpression(
      program.initial,
      inputType,
      unboundType(),
      unboundType(),
    );
    if (initialType.kind === "TypeError") return typeError();
    const reducerType = inferExpression(
      program.reducer,
      inputType,
      boundType(itemType.inferred),
      boundType(initialType.inferred),
    );
    if (
      reducerType.kind === "TypeOk" &&
      reducerType.inferred === initialType.inferred
    ) {
      return typeOk(initialType.inferred);
    }
    return typeError();
  }

  return inferExpression(program.body, inputType, unboundType(), unboundType());
}

//@ pure
export function evaluateMapInt(
  mapper: Expr,
  originalInput: RuntimeValue,
  list: IntList,
  mapperType: StaticType,
): EvalResult {
  //@ decreases list
  if (list.kind === "IntNil") {
    if (mapperType === "IntType") return evalOk(intListValue(intNil()));
    if (mapperType === "BoolType") return evalOk(boolListValue(boolNil()));
    return evalError();
  }
  const mappedHead = evaluateExpression(
    mapper,
    originalInput,
    boundValue(intValue(list.head)),
    unboundValue(),
  );
  if (mappedHead.kind === "EvalError") return evalError();
  const mappedTail = evaluateMapInt(mapper, originalInput, list.tail, mapperType);
  if (mappedTail.kind === "EvalError") return evalError();
  if (
    mapperType === "IntType" &&
    mappedHead.output.kind === "IntValue" &&
    mappedTail.output.kind === "IntListValue"
  ) {
    return evalOk(
      intListValue(
        intCons(mappedHead.output.intValue, mappedTail.output.intListValue),
      ),
    );
  }
  if (
    mapperType === "BoolType" &&
    mappedHead.output.kind === "BoolValue" &&
    mappedTail.output.kind === "BoolListValue"
  ) {
    return evalOk(
      boolListValue(
        boolCons(mappedHead.output.boolValue, mappedTail.output.boolListValue),
      ),
    );
  }
  return evalError();
}

//@ pure
export function evaluateMapBool(
  mapper: Expr,
  originalInput: RuntimeValue,
  list: BoolList,
  mapperType: StaticType,
): EvalResult {
  //@ decreases list
  if (list.kind === "BoolNil") {
    if (mapperType === "IntType") return evalOk(intListValue(intNil()));
    if (mapperType === "BoolType") return evalOk(boolListValue(boolNil()));
    return evalError();
  }
  const mappedHead = evaluateExpression(
    mapper,
    originalInput,
    boundValue(boolValue(list.head)),
    unboundValue(),
  );
  if (mappedHead.kind === "EvalError") return evalError();
  const mappedTail = evaluateMapBool(mapper, originalInput, list.tail, mapperType);
  if (mappedTail.kind === "EvalError") return evalError();
  if (
    mapperType === "IntType" &&
    mappedHead.output.kind === "IntValue" &&
    mappedTail.output.kind === "IntListValue"
  ) {
    return evalOk(
      intListValue(
        intCons(mappedHead.output.intValue, mappedTail.output.intListValue),
      ),
    );
  }
  if (
    mapperType === "BoolType" &&
    mappedHead.output.kind === "BoolValue" &&
    mappedTail.output.kind === "BoolListValue"
  ) {
    return evalOk(
      boolListValue(
        boolCons(mappedHead.output.boolValue, mappedTail.output.boolListValue),
      ),
    );
  }
  return evalError();
}

//@ pure
export function evaluateFoldRightInt(
  reducer: Expr,
  originalInput: RuntimeValue,
  list: IntList,
  initial: RuntimeValue,
): EvalResult {
  //@ decreases list
  if (list.kind === "IntNil") return evalOk(initial);
  const foldedTail = evaluateFoldRightInt(
    reducer,
    originalInput,
    list.tail,
    initial,
  );
  if (foldedTail.kind === "EvalError") return evalError();
  return evaluateExpression(
    reducer,
    originalInput,
    boundValue(intValue(list.head)),
    boundValue(foldedTail.output),
  );
}

//@ pure
export function evaluateFoldRightBool(
  reducer: Expr,
  originalInput: RuntimeValue,
  list: BoolList,
  initial: RuntimeValue,
): EvalResult {
  //@ decreases list
  if (list.kind === "BoolNil") return evalOk(initial);
  const foldedTail = evaluateFoldRightBool(
    reducer,
    originalInput,
    list.tail,
    initial,
  );
  if (foldedTail.kind === "EvalError") return evalError();
  return evaluateExpression(
    reducer,
    originalInput,
    boundValue(boolValue(list.head)),
    boundValue(foldedTail.output),
  );
}

//@ pure
export function evaluate(program: Program, input: RuntimeValue): EvalResult {
  if (program.kind === "MapProgram") {
    const inputType = valueType(input);
    const itemType = listItemType(inputType);
    if (itemType.kind === "TypeError") return evalError();
    const mapperType = inferExpression(
      program.mapper,
      inputType,
      boundType(itemType.inferred),
      unboundType(),
    );
    if (mapperType.kind === "TypeError") return evalError();
    if (input.kind === "IntListValue") {
      return evaluateMapInt(
        program.mapper,
        input,
        input.intListValue,
        mapperType.inferred,
      );
    }
    if (input.kind === "BoolListValue") {
      return evaluateMapBool(
        program.mapper,
        input,
        input.boolListValue,
        mapperType.inferred,
      );
    }
    return evalError();
  }

  if (program.kind === "FoldRightProgram") {
    const initial = evaluateExpression(
      program.initial,
      input,
      unboundValue(),
      unboundValue(),
    );
    if (initial.kind === "EvalError") return evalError();
    if (input.kind === "IntListValue") {
      return evaluateFoldRightInt(
        program.reducer,
        input,
        input.intListValue,
        initial.output,
      );
    }
    if (input.kind === "BoolListValue") {
      return evaluateFoldRightBool(
        program.reducer,
        input,
        input.boolListValue,
        initial.output,
      );
    }
    return evalError();
  }

  return evaluateExpression(
    program.body,
    input,
    unboundValue(),
    unboundValue(),
  );
}

//@ pure
export function expressionBodyCost(expr: Expr): bigint {
  //@ contract Return a structural expression-body cost that is always at least one.
  //@ ensures \result >= 1n
  if (
    expr.kind === "Input" ||
    expr.kind === "Item" ||
    expr.kind === "Accumulator" ||
    expr.kind === "IntLiteral" ||
    expr.kind === "BoolLiteral" ||
    expr.kind === "EmptyIntList" ||
    expr.kind === "EmptyBoolList"
  ) {
    return 1n;
  }
  if (expr.kind === "Not") {
    return 1n + expressionBodyCost(expr.operand);
  }
  if (expr.kind === "IfThenElse") {
    return (
      1n +
      expressionBodyCost(expr.condition) +
      expressionBodyCost(expr.thenExpr) +
      expressionBodyCost(expr.elseExpr)
    );
  }
  if (expr.kind === "PrependInt" || expr.kind === "PrependBool") {
    return 1n + expressionBodyCost(expr.head) + expressionBodyCost(expr.tail);
  }
  return 1n + expressionBodyCost(expr.left) + expressionBodyCost(expr.right);
}

//@ pure
export function expressionCost(program: Program): bigint {
  //@ contract Return a positive structural cost for a complete program.
  //@ ensures \result >= 1n
  if (program.kind === "MapProgram") {
    return 2n + expressionBodyCost(program.mapper);
  }
  if (program.kind === "FoldRightProgram") {
    return (
      3n +
      expressionBodyCost(program.initial) +
      expressionBodyCost(program.reducer)
    );
  }
  return expressionBodyCost(program.body);
}

//@ pure
export function matchesExample(program: Program, example: Example): boolean {
  const result = evaluate(program, example.input);
  if (result.kind === "EvalError") return false;
  return sameValue(result.output, example.output);
}

//@ pure
function examplesMatchSignatureFrom(
  examples: Example[],
  index: number,
  inputType: StaticType,
  outputType: StaticType,
): boolean {
  //@ requires index >= 0
  //@ decreases examples.length - index
  if (index >= examples.length) return true;
  const example = examples[index]!;
  if (
    valueType(example.input) !== inputType ||
    valueType(example.output) !== outputType
  ) {
    return false;
  }
  return examplesMatchSignatureFrom(examples, index + 1, inputType, outputType);
}

//@ pure
export function examplesHaveSignature(examples: Example[]): boolean {
  //@ contract If this checker accepts an example list, that list is nonempty.
  //@ ensures \result === true ==> examples.length > 0
  if (examples.length === 0) return false;
  return examplesMatchSignatureFrom(
    examples,
    1,
    valueType(examples[0]!.input),
    valueType(examples[0]!.output),
  );
}

//@ pure
function matchesAllExamplesFrom(
  program: Program,
  examples: Example[],
  index: number,
): boolean {
  //@ requires index >= 0
  //@ decreases examples.length - index
  if (index >= examples.length) return true;
  if (!matchesExample(program, examples[index]!)) return false;
  return matchesAllExamplesFrom(program, examples, index + 1);
}

//@ pure
export function matchesAllExamples(program: Program, examples: Example[]): boolean {
  return matchesAllExamplesFrom(program, examples, 0);
}

//@ pure
export function acceptProgram(program: Program, examples: Example[]): boolean {
  //@ contract If this checker accepts a program, its example list is nonempty.
  //@ ensures \result === true ==> examples.length > 0
  if (examples.length === 0) return false;
  if (!examplesHaveSignature(examples)) return false;
  const inferred = inferType(program, valueType(examples[0]!.input));
  if (inferred.kind === "TypeError") return false;
  if (inferred.inferred !== valueType(examples[0]!.output)) return false;
  return matchesAllExamples(program, examples);
}
