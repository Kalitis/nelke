import { useState } from "react";
import { Menu } from "@headlessui/react";
import { useChatStore } from "@/state/chatStore";
import type { ChatSummary } from "@/state/types";

function formatTime(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  if (sameDay) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function frontendLabel(frontend: string): string {
  return ({ telegram: "tg", tui: "tui", web: "web" }[frontend] ?? frontend) || "?";
}

function ChatRow({ chat }: { chat: ChatSummary }) {
  const activeChatId = useChatStore((s) => s.activeChatId);
  const selectChat = useChatStore((s) => s.selectChat);
  const renameChat = useChatStore((s) => s.renameChat);
  const deleteChat = useChatStore((s) => s.deleteChat);
  const active = chat.id === activeChatId;

  return (
    <div
      className={
        "group flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-2 text-sm transition-colors " +
        (active ? "bg-panel2 text-zinc-100" : "text-zinc-300 hover:bg-panel/60")
      }
      onClick={() => void selectChat(chat.id)}
    >
      <div className="min-w-0 flex-1">
        <div className="truncate">{chat.title || "New chat"}</div>
        <div className="truncate text-[11px] text-zinc-500">
          <span className="rounded bg-edge px-1 text-[10px] uppercase text-zinc-500">
            {frontendLabel(chat.frontend)}
          </span>{" "}
          {chat.message_count} msgs · {formatTime(chat.last_message_at || chat.started_at)}
        </div>
      </div>
      <Menu as="div" className="relative">
        <Menu.Button
          as="button"
          type="button"
          onClick={(e) => e.stopPropagation()}
          className="invisible rounded p-1 text-zinc-500 hover:bg-edge hover:text-zinc-100 group-hover:visible aria-expanded:visible"
          aria-label="Chat actions"
        >
          ⋯
        </Menu.Button>
        <Menu.Items
          as="div"
          className="absolute right-0 z-20 mt-1 w-40 rounded-lg border border-edge bg-panel2 p-1 shadow-xl"
        >
          <Menu.Item>
            {({ active: hover }) => (
              <button
                type="button"
                className={`block w-full rounded px-2 py-1.5 text-left text-sm ${
                  hover ? "bg-edge text-zinc-100" : "text-zinc-300"
                }`}
                onClick={(e) => {
                  e.stopPropagation();
                  const title = window.prompt("Rename chat:", chat.title);
                  if (title) void renameChat(chat.id, title);
                }}
              >
                Rename
              </button>
            )}
          </Menu.Item>
          <Menu.Item>
            {({ active: hover }) => (
              <button
                type="button"
                className={`block w-full rounded px-2 py-1.5 text-left text-sm ${
                  hover ? "bg-edge text-red-300" : "text-red-400"
                }`}
                onClick={(e) => {
                  e.stopPropagation();
                  if (window.confirm("Delete this chat and its history?")) {
                    void deleteChat(chat.id);
                  }
                }}
              >
                Delete
              </button>
            )}
          </Menu.Item>
        </Menu.Items>
      </Menu>
    </div>
  );
}

export function Sidebar({ onClose }: { onClose?: () => void }) {
  const chats = useChatStore((s) => s.chats);
  const createChat = useChatStore((s) => s.createChat);
  const [query, setQuery] = useState("");

  const filtered = query
    ? chats.filter((c) => (c.title || "New chat").toLowerCase().includes(query.toLowerCase()))
    : chats;

  return (
    <aside className="flex h-full w-72 flex-col border-r border-edge bg-panel">
      <div className="flex items-center gap-2 p-3">
        <button
          type="button"
          onClick={() => {
            void createChat();
            onClose?.();
          }}
          className="flex-1 rounded-lg border border-edge bg-panel2 px-3 py-2 text-sm font-medium text-zinc-100 transition-colors hover:bg-edge"
        >
          + New chat
        </button>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-2 text-zinc-400 hover:bg-panel2 lg:hidden"
            aria-label="Close sidebar"
          >
            ✕
          </button>
        )}
      </div>
      <div className="px-3 pb-2">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search chats…"
          className="w-full rounded-lg border border-edge bg-canvas px-3 py-1.5 text-sm placeholder:text-zinc-600 focus:border-accent focus:outline-none"
        />
      </div>
      <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
        {filtered.length === 0 ? (
          <p className="px-3 py-6 text-center text-sm text-zinc-600">No chats</p>
        ) : (
          filtered.map((c) => <ChatRow key={c.id} chat={c} />)
        )}
      </nav>
    </aside>
  );
}
