import { useEffect, useRef } from "react";
import { useChatStore, visibleTranscript } from "@/state/chatStore";
import { MessageBubble, StreamingBubble } from "./MessageBubble";
import type { ToolEntry } from "./ToolCallBlock";

export function Transcript() {
  const chat = useChatStore((s) => s.chat);
  const streaming = useChatStore((s) => s.streaming);
  const streamBuffer = useChatStore((s) => s.streamBuffer);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to the newest content while streaming.
  useEffect(() => {
    if (!bottomRef.current) return;
    bottomRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chat?.messages, streaming, streamBuffer.content]);

  const messages = visibleTranscript(chat);
  const tools: ToolEntry[] = streamBuffer.tools.map((t) => ({
    name: t.name,
    args: t.args,
    snippet: t.snippet,
  }));

  return (
    <div className="flex-1 overflow-y-auto">
      {messages.length === 0 && !streaming ? (
        <div className="mx-auto flex h-full max-w-3xl flex-col items-center justify-center px-4 text-center text-zinc-600">
          <div className="mb-2 text-4xl">✦</div>
          <p className="text-lg text-zinc-400">Start a conversation</p>
          <p className="mt-1 text-sm">
            Ask Nelke anything. Messages support markdown, code blocks, and branching.
          </p>
        </div>
      ) : (
        <div className="pb-40">
          {messages.map((m) => (
            <MessageBubble key={m.id} node={m} tree={chat!.tree} />
          ))}
          {streaming && (
            <StreamingBubble content={streamBuffer.content} tools={tools} />
          )}
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
