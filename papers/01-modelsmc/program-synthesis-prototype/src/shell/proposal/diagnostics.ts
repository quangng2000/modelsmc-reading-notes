import type { RuntimeValue } from "../../core/language.verify.js";

export function mismatchDiagnostic(predicted: RuntimeValue, expected: RuntimeValue): string {
  if (predicted.kind === "IntValue" && expected.kind === "IntValue") {
    return `expected-minus-predicted=${(expected.intValue - predicted.intValue).toString()}`;
  }
  if (predicted.kind === "BoolValue" && expected.kind === "BoolValue") {
    return `expected=${expected.boolValue}; predicted=${predicted.boolValue}`;
  }
  if (predicted.kind === "IntListValue" && expected.kind === "IntListValue") {
    let predictedTail = predicted.intListValue;
    let expectedTail = expected.intListValue;
    let index = 0;
    while (predictedTail.kind === "IntCons" && expectedTail.kind === "IntCons") {
      if (predictedTail.head !== expectedTail.head) {
        return `first mismatch at index ${index}: expected ${expectedTail.head.toString()}, predicted ${predictedTail.head.toString()}`;
      }
      predictedTail = predictedTail.tail;
      expectedTail = expectedTail.tail;
      index += 1;
    }
    if (predictedTail.kind === "IntCons") return `predicted list has extra items from index ${index}`;
    if (expectedTail.kind === "IntCons") return `predicted list is missing items from index ${index}`;
  }
  if (predicted.kind === "BoolListValue" && expected.kind === "BoolListValue") {
    let predictedTail = predicted.boolListValue;
    let expectedTail = expected.boolListValue;
    let index = 0;
    while (predictedTail.kind === "BoolCons" && expectedTail.kind === "BoolCons") {
      if (predictedTail.head !== expectedTail.head) {
        return `first mismatch at index ${index}: expected ${expectedTail.head}, predicted ${predictedTail.head}`;
      }
      predictedTail = predictedTail.tail;
      expectedTail = expectedTail.tail;
      index += 1;
    }
    if (predictedTail.kind === "BoolCons") return `predicted list has extra items from index ${index}`;
    if (expectedTail.kind === "BoolCons") return `predicted list is missing items from index ${index}`;
  }
  return "predicted value differs from expected value";
}
