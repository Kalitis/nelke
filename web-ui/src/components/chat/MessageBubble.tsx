import { useState } from "react";
import { useChatStore } from "@/state/chatStore";
import type { Message, MessageTree } from "@/state/types";
import { toolEntriesFromMessage } from "@/lib/tools";
import { Markdown } from "./Markdown";
import { MessageActions } from "./MessageActions";
import { SwipeNav } from "./SwipeNav";
import { EditComposer } from "./EditComposer";
import { ToolCallBlock, type ToolEntry } from "./ToolCallBlock";

interface MessageBubbleProps {
  node: Message;
  tree: MessageTree;
  // tool entries observed during the in-flight stream for an assistant bubble
  liveTools?: ToolEntry[];
}

function Avatar({ role }: { role: "user" | "assistant" }) {
  return (
    <div
      className={
        "flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-semibold " +
        (role === "assistant"
          ? "bg-accent/20 text-accent"
          : "bg-panel2 text-zinc-400")
      }
      aria-hidden="true"
    >
      {role === "assistant" ? "N" : "Y"}
    </div>
  );
}

export function MessageBubble({ node, tree, liveTools }: MessageBubbleProps) {
  const editMessage = useChatStore((s) => s.editMessage);
  const regenerateMessage = useChatStore((s) => s.regenerateMessage);
  const deleteMessage = useChatStore((s) => s.deleteMessage);
  const [editing, setEditing] = useState(false);

  const isUser = node.role === "user";
  // Tool-result rows are folded into their parent assistant's collapsible tool
  // block; rendering them standalone would dump the raw tool output into the
  // conversation as an assistant-looking bubble.
  if (node.role === "tool") return null;
  const persistedTools = isUser ? [] : toolEntriesFromMessage(node, tree);
  const tools = liveTools && liveTools.length ? liveTools : persistedTools;

  if (editing && isUser) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-3">
        <EditComposer
          initial={node.content}
          onSave={(text) => {
            setEditing(false);
            void editMessage(node.id, text);
          }}
          onCancel={() => setEditing(false)}
        />
      </div>
    );
  }

  return (
    <div className="group mx-auto max-w-3xl px-4 py-4">
      <div className="flex gap-3">
        <Avatar role={isUser ? "user" : "assistant"} />
        <div className="min-w-0 flex-1">
          <div className="mb-1 text-xs font-medium text-zinc-500">
            {isUser ? "You" : "Nelke"}
          </div>
          {isUser ? (
            <div className="whitespace-pre-wrap break-words text-zinc-100">{node.content}</div>
          ) : node.content || tools.length ? (
            <>
              {node.content && <Markdown content={node.content} />}
              {tools.map((t, i) => (
                <ToolCallBlock key={`${node.id}-tool-${i}`} tool={t} />
              ))}
            </>
          ) : (
            <div className="text-zinc-600">…</div>
          )}
          {!isUser && <SwipeNav node={node} tree={tree} />}
          <MessageActions
            role={isUser ? "user" : "assistant"}
            content={node.content}
            onEdit={() => setEditing(true)}
            onRegenerate={() => void regenerateMessage(node.id)}
            onDelete={() => {
              if (window.confirm("Delete this message and everything after it?")) {
                void deleteMessage(node.id);
              }
            }}
          />
        </div>
      </div>
    </div>
  );
}

/** Placeholder bubble used while streaming an answer that isn't persisted yet. */
export function StreamingBubble({
  content,
  tools,
}: {
  content: string;
  tools: ToolEntry[];
}) {
  return (
    <div className="mx-auto max-w-3xl px-4 py-4">
      <div className="flex gap-3">
        <Avatar role="assistant" />
        <div className="min-w-0 flex-1">
          <div className="mb-1 text-xs font-medium text-zinc-500">Nelke</div>
          {tools.map((t, i) => (
            <ToolCallBlock key={i} tool={t} />
          ))}
          {content ? (
            <Markdown content={content} />
          ) : (
            <div className="flex items-center gap-2 text-zinc-600">
              <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-edge border-t-accent" />
              thinking…
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
