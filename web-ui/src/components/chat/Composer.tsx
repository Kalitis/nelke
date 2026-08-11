import { useEffect, useRef, useState } from "react";
import { useChatStore } from "@/state/chatStore";

export function Composer() {
  const activeChatId = useChatStore((s) => s.activeChatId);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const streaming = useChatStore((s) => s.streaming);
  const createChat = useChatStore((s) => s.createChat);
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  // Autosize the textarea up to a sane cap.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 240) + "px";
  }, [text]);

  const submit = () => {
    const value = text.trim();
    if (!value || streaming) return;
    if (!activeChatId) {
      void createChat().then(() => sendMessage(value));
    } else {
      void sendMessage(value);
    }
    setText("");
  };

  return (
    <div className="border-t border-edge bg-canvas/80 px-4 py-3 backdrop-blur">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-2xl border border-edge bg-panel2 p-2 focus-within:border-accent">
          <textarea
            ref={ref}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder={activeChatId ? "Message Nelke…" : "Start a new chat…"}
            className="max-h-60 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none"
          />
          <button
            type="button"
            onClick={submit}
            disabled={!text.trim() || streaming}
            className="rounded-xl bg-accent px-3.5 py-2 text-sm font-medium text-canvas transition-colors hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Send
          </button>
        </div>
        <p className="mt-1.5 text-center text-[11px] text-zinc-600">
          Enter to send · Shift+Enter for newline · Nelke can make mistakes.
        </p>
      </div>
    </div>
  );
}
