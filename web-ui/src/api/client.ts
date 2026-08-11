import type {
  ChatDetail,
  ChatSummary,
  Message,
  MessageTree,
  Profile,
  StreamEvent,
} from "@/state/types";

async function json<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status}: ${text || resp.statusText}`);
  }
  return (await resp.json()) as T;
}

// ---- REST -----------------------------------------------------------------

export const api = {
  async health(): Promise<{ ok: boolean }> {
    return json(await fetch("/api/health"));
  },

  async profiles(): Promise<Profile[]> {
    return json(await fetch("/api/profiles"));
  },

  async listChats(): Promise<ChatSummary[]> {
    return json(await fetch("/api/chats"));
  },

  async createChat(title?: string): Promise<{ id: string; title: string }> {
    return json(
      await fetch("/api/chats", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title ?? null }),
      }),
    );
  },

  async getChat(chatId: string): Promise<ChatDetail> {
    return json(await fetch(`/api/chats/${encodeURIComponent(chatId)}`));
  },

  async renameChat(chatId: string, title: string): Promise<{ ok: boolean }> {
    return json(
      await fetch(`/api/chats/${encodeURIComponent(chatId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      }),
    );
  },

  async deleteChat(chatId: string): Promise<{ ok: boolean }> {
    return json(
      await fetch(`/api/chats/${encodeURIComponent(chatId)}`, {
        method: "DELETE",
      }),
    );
  },

  async getTree(chatId: string): Promise<{
    nodes: Record<string, Message>;
    children: MessageTree["children"];
    root_id: string | null;
    active_leaf_id: string | null;
  }> {
    return json(await fetch(`/api/chats/${encodeURIComponent(chatId)}/tree`));
  },

  async editMessage(
    chatId: string,
    messageId: string,
    content: string,
  ): Promise<{ message_id: string; parent_id: string | null }> {
    return json(
      await fetch(
        `/api/chats/${encodeURIComponent(chatId)}/messages/${encodeURIComponent(messageId)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content }),
        },
      ),
    );
  },

  async deleteMessage(
    chatId: string,
    messageId: string,
  ): Promise<{ deleted_id: string; active_leaf_id: string | null }> {
    return json(
      await fetch(
        `/api/chats/${encodeURIComponent(chatId)}/messages/${encodeURIComponent(messageId)}`,
        { method: "DELETE" },
      ),
    );
  },

  async activateMessage(
    chatId: string,
    messageId: string,
  ): Promise<{
    active_leaf_id: string | null;
    tree: MessageTree;
    messages: Message[];
  }> {
    return json(
      await fetch(
        `/api/chats/${encodeURIComponent(chatId)}/messages/${encodeURIComponent(messageId)}/activate`,
        { method: "POST" },
      ),
    );
  },
};

// ---- Streaming (SSE over fetch ReadableStream) ---------------------------
//
// EventSource only supports GET, so the chat / regenerate endpoints are
// consumed via fetch + manual SSE frame parsing (same approach as the legacy
// vanilla client, but typed and reusable).

export interface StreamHandlers {
  onEvent: (ev: StreamEvent) => void;
  onError?: (err: Error) => void;
}

interface StreamBody {
  text: string;
  profile?: string | null;
  parent_message_id?: string | null;
}

async function pumpStream(resp: Response, handlers: StreamHandlers): Promise<void> {
  if (!resp.ok || !resp.body) {
    throw new Error(`HTTP ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by blank lines.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const ev = parseFrame(frame);
      if (ev) handlers.onEvent(ev);
    }
  }
}

function parseFrame(frame: string): StreamEvent | null {
  let event = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data += line.slice(5).trim();
    }
  }
  if (!data) return null;
  try {
    const parsed = JSON.parse(data);
    return { event, data: parsed } as StreamEvent;
  } catch {
    return { event: "token", data: { text: data } };
  }
}

export async function streamChatMessage(
  chatId: string,
  body: StreamBody,
  handlers: StreamHandlers,
): Promise<void> {
  try {
    const resp = await fetch(
      `/api/chats/${encodeURIComponent(chatId)}/messages`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );
    await pumpStream(resp, handlers);
  } catch (err) {
    handlers.onError?.(err as Error);
  }
}

export async function streamRegenerate(
  chatId: string,
  messageId: string,
  profile: string | null,
  handlers: StreamHandlers,
): Promise<void> {
  try {
    const resp = await fetch(
      `/api/chats/${encodeURIComponent(chatId)}/messages/${encodeURIComponent(messageId)}/regenerate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile }),
      },
    );
    await pumpStream(resp, handlers);
  } catch (err) {
    handlers.onError?.(err as Error);
  }
}
