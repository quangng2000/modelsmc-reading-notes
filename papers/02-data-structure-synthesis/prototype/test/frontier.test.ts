import assert from "node:assert/strict";
import test from "node:test";

import {
  emptyFrontier,
  popMinFrontier,
  pushFrontier,
} from "../src/frontier.js";

test("pops least cost first and preserves insertion order for ties", () => {
  let frontier = emptyFrontier<string>();
  frontier = pushFrontier(frontier, "four", 4);
  frontier = pushFrontier(frontier, "one-a", 1);
  frontier = pushFrontier(frontier, "three", 3);
  frontier = pushFrontier(frontier, "one-b", 1);

  const popped: string[] = [];
  const costs: number[] = [];

  while (true) {
    const result = popMinFrontier(frontier);
    if (result === undefined) {
      break;
    }
    popped.push(result.item);
    costs.push(result.cost);
    frontier = result.frontier;
  }

  assert.deepEqual(popped, ["one-a", "one-b", "three", "four"]);
  assert.deepEqual(costs, [1, 1, 3, 4]);
});

test("push and pop preserve earlier frontier versions", () => {
  const empty = emptyFrontier<string>();
  const withTwo = pushFrontier(empty, "two", 2);
  const withOneAndTwo = pushFrontier(withTwo, "one", 1);

  assert.equal(empty.size, 0);
  assert.equal(withTwo.size, 1);
  assert.equal(popMinFrontier(withTwo)?.item, "two");
  assert.equal(popMinFrontier(withOneAndTwo)?.item, "one");
  assert.equal(popMinFrontier(withTwo)?.item, "two");
});
