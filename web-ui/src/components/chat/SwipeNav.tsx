import { useChatStore } from "@/state/chatStore";
import { childrenOf, siblingPosition } from "@/lib/tree";
import type { Message, MessageTree } from "@/state/types";

interface SwipeNavProps {
  node: Message;
  tree: MessageTree;
}

/** ‹ 2/3 › navigator for sibling alternatives of an assistant turn. */
export function SwipeNav({ node, tree }: SwipeNavProps) {
  const swipeTo = useChatStore((s) => s.swipeTo);
  const streaming = useChatStore((s) => s.streaming);
  const siblings = childrenOf(tree, node.parent_id).filter((s) => s.role === "assistant");
  if (siblings.length <= 1) return null;
  const { index, total } = siblingPosition(tree, node);
  const prev = siblings[index - 1];
  const next = siblings[index + 1];

  return (
    <div className="mt-1 flex items-center gap-1 text-xs text-zinc-500">
      <button
        type="button"
        disabled={!prev || streaming}
        onClick={() => prev && void swipeTo(prev.id)}
        className="rounded px-1.5 py-0.5 hover:bg-panel2 disabled:opacity-30"
        aria-label="Previous response"
      >
        ‹
      </button>
      <span className="tabular-nums">
        {index + 1} / {total}
      </span>
      <button
        type="button"
        disabled={!next || streaming}
        onClick={() => next && void swipeTo(next.id)}
        className="rounded px-1.5 py-0.5 hover:bg-panel2 disabled:opacity-30"
        aria-label="Next response"
      >
        ›
      </button>
    </div>
  );
}
