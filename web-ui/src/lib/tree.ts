import type { Message, MessageTree } from "@/state/types";

/** Build the root→leaf active path the transcript renders. */
export function activePath(tree: MessageTree): Message[] {
  const leafId = findActiveLeaf(tree);
  if (!leafId) return [];
  return pathToRoot(tree, leafId).reverse();
}

/** Walk from a node up to the root collecting the chain (leaf→root order). */
export function pathToRoot(tree: MessageTree, nodeId: string): Message[] {
  const out: Message[] = [];
  let current: string | null = nodeId;
  const seen = new Set<string>();
  while (current && !seen.has(current)) {
    seen.add(current);
    const node: Message | undefined = tree.nodes[current];
    if (!node) break;
    out.push(node);
    current = node.parent_id;
  }
  return out;
}

/**
 * Find the deepest active node (the active path's leaf). A node is a leaf if
 * it has no active, non-deleted children.
 */
export function findActiveLeaf(tree: MessageTree): string | null {
  for (const id of Object.keys(tree.nodes)) {
    const node = tree.nodes[id];
    if (!node.is_active || node.is_deleted) continue;
    const kids = childrenOf(tree, id).filter((c) => c.is_active && !c.is_deleted);
    if (kids.length === 0) return id;
  }
  return null;
}

/** Direct, non-deleted children of a parent (swipe alternatives). */
export function childrenOf(tree: MessageTree, parentId: string | null): Message[] {
  const key = parentId ?? "null";
  return (tree.children[key] ?? []).filter((c) => !c.is_deleted);
}

/** Index of a node among its siblings + total sibling count (for `‹ 2/3 ›`). */
export function siblingPosition(
  tree: MessageTree,
  node: Message,
): { index: number; total: number } {
  const siblings = childrenOf(tree, node.parent_id);
  const index = siblings.findIndex((s) => s.id === node.id);
  return { index: Math.max(0, index), total: siblings.length };
}

/** Count active, non-deleted messages in a tree (transcript length). */
export function countActive(tree: MessageTree): number {
  return Object.values(tree.nodes).filter((n) => n.is_active && !n.is_deleted).length;
}
