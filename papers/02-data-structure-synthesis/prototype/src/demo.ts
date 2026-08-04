import { renderExpression } from "./ast.js";
import { enumerateExpressionsByCost, synthesize } from "./enumerator.js";
import { synthesizeMap } from "./synthesizer.js";

console.log("Scalar candidates, cheapest first:");
for (const bucket of enumerateExpressionsByCost({ maxCost: 2 })) {
  const preview = bucket.expressions.slice(0, 8).map(renderExpression);
  const suffix = bucket.expressions.length > preview.length ? ", ..." : "";
  console.log(`cost ${bucket.cost}: ${preview.join(", ")}${suffix}`);
}

const scalar = synthesize(
  [
    { input: 1, output: 3 },
    { input: 2, output: 4 },
  ],
  { maxCost: 4 },
);

if (scalar !== undefined) {
  console.log(
    `Scalar solution: ${renderExpression(scalar.expression)} (cost ${scalar.cost})`,
  );
}

const mapped = synthesizeMap(
  [
    { input: [1, 2], output: [3, 4] },
    { input: [3], output: [5] },
  ],
  { maxCost: 4 },
);

if (mapped.kind === "synthesized") {
  const inferred = mapped.inferredExamples
    .map(({ input, output }) => `${input} -> ${output}`)
    .join(", ");
  console.log(`Deduction inferred: ${inferred}`);
  console.log(`Completed program: ${renderExpression(mapped.program)}`);
  console.log(
    `Frontier pops: ${mapped.trace.map(({ cost, count }) => `${cost} × ${count}`).join(" -> ")}`,
  );
}
