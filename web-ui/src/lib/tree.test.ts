import { describe, expect, it } from "vitest";
import {
  activePath,
  appendOptimistic,
  childrenOf,
  countActive,
  findActiveLeaf,
  pathToRoot,
  removeOptimistic,
  siblingPosition,
} from "./tree";
import type { Message, MessageTree } from "@/state/types";

let n = 0;
function mk(
  role: Message["role"],
  parent_id: string | null,
  opts: Partial<Message> = {},
): Message {
  n += 1;
  return {
    id: `m${n}`,
    role,
    content: opts.content ?? `msg-${n}`,
    parent_id,
    is_active: opts.is_active ?? true,
    is_deleted: opts.is_deleted ?? false,
    sibling_order: opts.sibling_order ?? n,
  };
}

function build(nodes: Message[]): MessageTree {
  const tree: MessageTree = { root_id: null, nodes: {}, children: {} };
  for (const node of nodes) {
    tree.nodes[node.id] = node;
    const key = node.parent_id ?? "null";
    (tree.children[key] ??= []).push(node);
    if (node.parent_id === null && tree.root_id === null) tree.root_id = node.id;
  }
  return tree;
}

describe("tree helpers", () => {
  it("walks a linear chain from leaf to root", () => {
    const a = mk("user", null);
    const b = mk("assistant", a.id);
    const c = mk("user", b.id);
    const tree = build([a, b, c]);
    expect(pathToRoot(tree, c.id).map((m) => m.id)).toEqual([c.id, b.id, a.id]);
  });

  it("finds the single active leaf", () => {
    const a = mk("user", null);
    const b = mk("assistant", a.id);
    const tree = build([a, b]);
    expect(findActiveLeaf(tree)).toBe(b.id);
  });

  it("activePath returns root→leaf in order", () => {
    const a = mk("user", null);
    const b = mk("assistant", a.id);
    const c = mk("user", b.id);
    const tree = build([a, b, c]);
    expect(activePath(tree).map((m) => m.id)).toEqual([a.id, b.id, c.id]);
  });

  it("childrenOf returns non-deleted siblings and ignores other branches", () => {
    const u = mk("user", null);
    const a1 = mk("assistant", u.id, { sibling_order: 0 });
    const a2 = mk("assistant", u.id, { sibling_order: 1 });
    const a3 = mk("assistant", u.id, { is_deleted: true, sibling_order: 2 });
    const tree = build([u, a1, a2, a3]);
    const kids = childrenOf(tree, u.id);
    expect(kids.map((k) => k.id)).toEqual([a1.id, a2.id]);
  });

  it("siblingPosition gives 1-based index among live siblings", () => {
    const u = mk("user", null);
    const a1 = mk("assistant", u.id, { sibling_order: 0 });
    const a2 = mk("assistant", u.id, { sibling_order: 1 });
    const a3 = mk("assistant", u.id, { sibling_order: 2 });
    const tree = build([u, a1, a2, a3]);
    expect(siblingPosition(tree, a2)).toEqual({ index: 1, total: 3 });
  });

  it("countActive skips inactive/deleted nodes", () => {
    const a = mk("user", null);
    const b = mk("assistant", a.id);
    const c = mk("assistant", a.id, { is_active: false });
    const d = mk("user", null, { is_deleted: true });
    const tree = build([a, b, c, d]);
    expect(countActive(tree)).toBe(2);
  });
});

describe("appendOptimistic / removeOptimistic", () => {
  it("appendOptimistic adds a new leaf under the given parent and makes it the active leaf", () => {
    const a = mk("user", null);
    const b = mk("assistant", a.id);
    const tree = build([a, b]);
    const optimistic = mk("user", b.id);
    optimistic.id = "pending-1";

    const next = appendOptimistic(tree, optimistic);

    // Original tree is untouched (immutability).
    expect(tree.nodes).not.toHaveProperty("pending-1");
    // New tree contains the node, parented correctly, and as a child of b.
    expect(next.nodes).toHaveProperty("pending-1");
    expect(next.nodes["pending-1"].parent_id).toBe(b.id);
    expect(childrenOf(next, b.id).map((m) => m.id)).toContain("pending-1");
    // New node becomes the active leaf so visibleTranscript surfaces it.
    expect(findActiveLeaf(next)).toBe("pending-1");
    expect(activePath(next).map((m) => m.id)).toEqual([a.id, b.id, "pending-1"]);
  });

  it("appendOptimistic on an empty tree sets it as the root", () => {
    const empty: MessageTree = { root_id: null, nodes: {}, children: {} };
    const optimistic = mk("user", null);
    optimistic.id = "pending-1";

    const next = appendOptimistic(empty, optimistic);

    expect(next.root_id).toBe("pending-1");
    expect(activePath(next).map((m) => m.id)).toEqual(["pending-1"]);
  });

  it("removeOptimistic drops the node and cleans up its parent's child list", () => {
    const a = mk("user", null);
    const b = mk("assistant", a.id);
    const tree = build([a, b]);
    const optimistic = mk("user", b.id);
    optimistic.id = "pending-1";
    const withOpt = appendOptimistic(tree, optimistic);

    const after = removeOptimistic(withOpt, "pending-1");

    expect(after.nodes).not.toHaveProperty("pending-1");
    expect(childrenOf(after, b.id).map((m) => m.id)).toEqual([]);
    // Original active leaf is restored.
    expect(findActiveLeaf(after)).toBe(b.id);
  });

  it("removeOptimistic on a missing id is a no-op", () => {
    const a = mk("user", null);
    const tree = build([a]);
    expect(removeOptimistic(tree, "nope")).toBe(tree);
  });
});
