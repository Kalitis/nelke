import { create } from "zustand";
import { api, streamChatMessage, streamRegenerate } from "@/api/client";
import type {
  ChatDetail,
  ChatSummary,
  Message,
  MessageTree,
  Profile,
  StreamEvent,
} from "@/state/types";
import { activePath, findActiveLeaf } from "@/lib/tree";
import { pathToRoot } from "@/lib/tree";

interface ChatState {
  chats: ChatSummary[];
  activeChatId: string | null;
  chat: ChatDetail | null;
  profiles: Profile[];
  profile: string | null;
  streaming: boolean;
  // transient streaming buffer surfaced to the UI while a turn is in flight
  streamBuffer: { content: string; tools: StreamToolEntry[] };
  error: string | null;

  loadProfiles: () => Promise<void>;
  loadChats: () => Promise<void>;
  selectChat: (id: string) => Promise<void>;
  createChat: () => Promise<void>;
  renameChat: (id: string, title: string) => Promise<void>;
  deleteChat: (id: string) => Promise<void>;
  setProfile: (name: string) => void;

  sendMessage: (text: string) => Promise<void>;
  editMessage: (messageId: string, content: string) => Promise<void>;
  regenerateMessage: (messageId: string) => Promise<void>;
  deleteMessage: (messageId: string) => Promise<void>;
  swipeTo: (messageId: string) => Promise<void>;
}

interface StreamToolEntry {
  name: string;
  args: Record<string, unknown>;
  snippet?: string;
}

function emptyTree(): MessageTree {
  return { root_id: null, nodes: {}, children: {} };
}

export const useChatStore = create<ChatState>((set, get) => ({
  chats: [],
  activeChatId: null,
  chat: null,
  profiles: [],
  profile: null,
  streaming: false,
  streamBuffer: { content: "", tools: [] },
  error: null,

  loadProfiles: async () => {
    try {
      const profiles = await api.profiles();
      set({ profiles, profile: get().profile ?? profiles[0]?.name ?? null });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  loadChats: async () => {
    try {
      const chats = await api.listChats();
      set({ chats });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  selectChat: async (id) => {
    set({ activeChatId: id, error: null, streamBuffer: { content: "", tools: [] } });
    try {
      const chat = await api.getChat(id);
      set({ chat });
    } catch (err) {
      set({ error: String(err), chat: null });
    }
  },

  createChat: async () => {
    try {
      const { id } = await api.createChat();
      await get().loadChats();
      await get().selectChat(id);
    } catch (err) {
      set({ error: String(err) });
    }
  },

  renameChat: async (id, title) => {
    try {
      await api.renameChat(id, title);
      await get().loadChats();
      if (get().activeChatId === id && get().chat) {
        set({ chat: { ...get().chat!, title } });
      }
    } catch (err) {
      set({ error: String(err) });
    }
  },

  deleteChat: async (id) => {
    try {
      await api.deleteChat(id);
      await get().loadChats();
      if (get().activeChatId === id) {
        set({ activeChatId: null, chat: null });
      }
    } catch (err) {
      set({ error: String(err) });
    }
  },

  setProfile: (name) => set({ profile: name }),

  sendMessage: async (text) => {
    const { activeChatId, profile, chat } = get();
    if (!activeChatId || !text.trim() || get().streaming) return;
    // Anchor the new turn to the current active leaf so branching keeps working.
    const parentId = chat ? findActiveLeaf(chat.tree) : null;
    await runStream(
      set,
      () =>
        streamChatMessage(
          activeChatId,
          { text, profile, parent_message_id: parentId },
          makeHandlers(set),
        ),
    );
  },

  editMessage: async (messageId, content) => {
    const { activeChatId } = get();
    if (!activeChatId) return;
    try {
      const result = await api.editMessage(activeChatId, messageId, content);
      // Reload the tree so the new sibling is visible, then stream a fresh
      // assistant answer from the edited user message.
      const fresh = await api.getChat(activeChatId);
      set({ chat: fresh });
      await runStream(
        set,
        () =>
          streamChatMessage(
            activeChatId,
            { text: content, profile: get().profile, parent_message_id: result.message_id },
            makeHandlers(set),
          ),
      );
    } catch (err) {
      set({ error: String(err) });
    }
  },

  regenerateMessage: async (messageId) => {
    const { activeChatId, profile } = get();
    if (!activeChatId) return;
    await runStream(set, () =>
      streamRegenerate(activeChatId, messageId, profile, makeHandlers(set)),
    );
  },

  deleteMessage: async (messageId) => {
    const { activeChatId } = get();
    if (!activeChatId) return;
    try {
      await api.deleteMessage(activeChatId, messageId);
      const fresh = await api.getChat(activeChatId);
      set({ chat: fresh });
      await get().loadChats();
    } catch (err) {
      set({ error: String(err) });
    }
  },

  swipeTo: async (messageId) => {
    const { activeChatId } = get();
    if (!activeChatId) return;
    try {
      const result = await api.activateMessage(activeChatId, messageId);
      if (get().chat) {
        set({
          chat: { ...get().chat!, tree: result.tree, messages: result.messages },
        });
      }
    } catch (err) {
      set({ error: String(err) });
    }
  },
}));

// ---- streaming helper ----------------------------------------------------

function makeHandlers(
  set: (partial: Partial<ChatState>) => void,
): { onEvent: (ev: StreamEvent) => void; onError: (err: Error) => void } {
  return {
    onEvent: (ev) => {
      const buf = useChatStore.getState().streamBuffer;
      switch (ev.event) {
        case "token":
          set({ streamBuffer: { ...buf, content: buf.content + ev.data.text } });
          break;
        case "tool":
          set({
            streamBuffer: {
              ...buf,
              tools: [...buf.tools, { name: ev.data.name, args: ev.data.args }],
            },
          });
          break;
        case "tool_result":
          set({
            streamBuffer: {
              ...buf,
              tools: buf.tools.map((t, i) =>
                i === buf.tools.length - 1 && !t.snippet
                  ? { ...t, snippet: ev.data.snippet }
                  : t,
              ),
            },
          });
          break;
        case "error":
          set({ error: ev.data.message, streaming: false });
          break;
        case "done":
          // Server has persisted the new messages; refresh the canonical view.
          void useChatStore.getState().selectChat(useChatStore.getState().activeChatId!);
          void useChatStore.getState().loadChats();
          break;
        default:
          break;
      }
    },
    onError: (err) => set({ error: String(err), streaming: false }),
  };
}

async function runStream(
  set: (partial: Partial<ChatState>) => void,
  run: () => Promise<void>,
): Promise<void> {
  set({ streaming: true, error: null, streamBuffer: { content: "", tools: [] } });
  try {
    await run();
  } finally {
    set({ streaming: false });
  }
}

// Re-export for components that want to derive the visible transcript.
export function visibleTranscript(chat: ChatDetail | null): Message[] {
  if (!chat) return [];
  return activePath(chat.tree);
}

export { emptyTree, pathToRoot };
