import assert from "node:assert/strict";
import test from "node:test";

import {
  effectiveSampleSize,
  normalizeLogWeights,
} from "../src/shell/engine/numerics.js";
import { SeededRandom } from "../src/shell/engine/random.js";
import { systematicResample } from "../src/shell/engine/resampling.js";

test("normalizes log weights and computes ESS and systematic ancestors deterministically", () => {
  const normalized = normalizeLogWeights([Math.log(1), Math.log(2), Math.log(1)]);

  assert.equal(normalized.usedUniformFallback, false);
  assert.deepEqual(normalized.weights, [0.25, 0.5, 0.25]);
  assert.ok(Math.abs(effectiveSampleSize(normalized.weights) - 8 / 3) < 1e-12);
  assert.deepEqual(systematicResample(normalized.weights, { next: () => 0 }), [0, 1, 1]);

  const first = systematicResample(normalized.weights, new SeededRandom(41));
  const second = systematicResample(normalized.weights, new SeededRandom(41));
  assert.deepEqual(first, second);
});
