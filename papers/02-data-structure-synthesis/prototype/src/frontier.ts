interface FrontierEntry<T> {
  readonly item: T;
  readonly cost: number;
  readonly insertionOrder: number;
}

interface HeapNode<T> {
  readonly entry: FrontierEntry<T>;
  readonly left: HeapNode<T> | undefined;
  readonly right: HeapNode<T> | undefined;
  readonly rank: number;
}

export interface Frontier<T> {
  readonly root: HeapNode<T> | undefined;
  readonly nextInsertionOrder: number;
  readonly size: number;
}

export interface PopResult<T> {
  readonly item: T;
  readonly cost: number;
  readonly frontier: Frontier<T>;
}

export function emptyFrontier<T>(): Frontier<T> {
  return { root: undefined, nextInsertionOrder: 0, size: 0 };
}

export function pushFrontier<T>(
  frontier: Frontier<T>,
  item: T,
  cost: number,
): Frontier<T> {
  if (!Number.isSafeInteger(cost) || cost < 0) {
    throw new Error("Frontier cost must be a nonnegative safe integer.");
  }
  if (!Number.isSafeInteger(frontier.nextInsertionOrder)) {
    throw new Error("Frontier insertion order exceeded the safe-integer range.");
  }

  const singleton: HeapNode<T> = {
    entry: {
      item,
      cost,
      insertionOrder: frontier.nextInsertionOrder,
    },
    left: undefined,
    right: undefined,
    rank: 1,
  };

  return {
    root: merge(frontier.root, singleton),
    nextInsertionOrder: frontier.nextInsertionOrder + 1,
    size: frontier.size + 1,
  };
}

export function popMinFrontier<T>(
  frontier: Frontier<T>,
): PopResult<T> | undefined {
  const root = frontier.root;
  if (root === undefined) {
    return undefined;
  }

  return {
    item: root.entry.item,
    cost: root.entry.cost,
    frontier: {
      root: merge(root.left, root.right),
      nextInsertionOrder: frontier.nextInsertionOrder,
      size: frontier.size - 1,
    },
  };
}

function merge<T>(
  left: HeapNode<T> | undefined,
  right: HeapNode<T> | undefined,
): HeapNode<T> | undefined {
  if (left === undefined) {
    return right;
  }
  if (right === undefined) {
    return left;
  }

  if (comesBefore(right.entry, left.entry)) {
    return mergeRoots(right, left);
  }
  return mergeRoots(left, right);
}

function mergeRoots<T>(first: HeapNode<T>, second: HeapNode<T>): HeapNode<T> {
  return makeNode(first.entry, first.left, merge(first.right, second));
}

function makeNode<T>(
  entry: FrontierEntry<T>,
  left: HeapNode<T> | undefined,
  right: HeapNode<T> | undefined,
): HeapNode<T> {
  if (rank(left) < rank(right)) {
    return {
      entry,
      left: right,
      right: left,
      rank: rank(left) + 1,
    };
  }

  return {
    entry,
    left,
    right,
    rank: rank(right) + 1,
  };
}

function rank<T>(node: HeapNode<T> | undefined): number {
  return node?.rank ?? 0;
}

function comesBefore<T>(
  left: FrontierEntry<T>,
  right: FrontierEntry<T>,
): boolean {
  return (
    left.cost < right.cost ||
    (left.cost === right.cost &&
      left.insertionOrder < right.insertionOrder)
  );
}
