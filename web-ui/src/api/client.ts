import type {
  ChatDetail,
  ChatSummary,
  CycleDetail,
  CycleSummary,
  Message,
  MessageTree,
  MemoryFile,
  Profile,
  ProjectDetail,
  ProjectSummary,
  StreamEvent,
  UsageAggregate,
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

  // ---- cycles / memory (secondary SPA screens) ---------------------------

  async cyclesList(): Promise<CycleSummary[]> {
    return json(await fetch("/api/cycles/list"));
  },

  async cycleDetail(cycleId: string): Promise<CycleDetail> {
    return json(await fetch(`/api/cycles/${encodeURIComponent(cycleId)}`));
  },

  async startCycle(
    objective: string,
    autoApprove: boolean,
  ): Promise<{ status: string; cycle_id?: string }> {
    return json(
      await fetch("/api/improve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ objective, auto_approve: autoApprove }),
      }),
    );
  },

  async memoryFiles(): Promise<MemoryFile[]> {
    return json(await fetch("/api/memory"));
  },

  async memoryFile(name: string): Promise<{ name: string; content: string }> {
    return json(await fetch(`/api/memory/${name.split("/").map(encodeURIComponent).join("/")}`));
  },

  // ---- projects (group chats + per-project memory) -----------------------

  async projectsList(): Promise<ProjectSummary[]> {
    return json(await fetch("/api/projects"));
  },

  async projectDetail(projectId: string): Promise<ProjectDetail> {
    return json(await fetch(`/api/projects/${encodeURIComponent(projectId)}`));
  },

  async createProject(
    name: string,
    opts?: { description?: string; stage?: string },
  ): Promise<{ id: string; name: string }> {
    return json(
      await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          description: opts?.description ?? "",
          stage: opts?.stage ?? "",
        }),
      }),
    );
  },

  async updateProject(
    projectId: string,
    fields: { name?: string; description?: string; stage?: string },
  ): Promise<{ ok: boolean }> {
    return json(
      await fetch(`/api/projects/${encodeURIComponent(projectId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fields),
      }),
    );
  },

  async deleteProject(projectId: string): Promise<{ ok: boolean }> {
    return json(
      await fetch(`/api/projects/${encodeURIComponent(projectId)}`, { method: "DELETE" }),
    );
  },

  async linkChat(projectId: string, chatId: string): Promise<{ ok: boolean }> {
    return json(
      await fetch(
        `/api/projects/${encodeURIComponent(projectId)}/chats/${encodeURIComponent(chatId)}`,
        { method: "POST" },
      ),
    );
  },

  async setProjectMemory(
    projectId: string,
    name: string,
    content: string,
    append = false,
  ): Promise<{ ok: boolean; name: string }> {
    return json(
      await fetch(`/api/projects/${encodeURIComponent(projectId)}/memory`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, content, append }),
      }),
    );
  },

  // DB-backed token usage for a chat (totals + recent per-call events).
  async usage(sessionId: string): Promise<UsageAggregate> {
    return json(
      await fetch(`/api/usage?session_id=${encodeURIComponent(sessionId)}`),
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

export async function pumpStream(resp: Response, handlers: StreamHandlers): Promise<void> {
  if (!resp.ok || !resp.body) {
    throw new Error(`HTTP ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const flush = () => {
    if (!buffer.trim()) return;
    const ev = parseFrame(buffer);
    if (ev) handlers.onEvent(ev);
    buffer = "";
  };
  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      // sse_starlette may close the stream immediately after the final frame
      // without a trailing blank-line separator. Flush whatever remains so the
      // terminal `done`/`cycle_result` event is not silently dropped.
      flush();
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line. sse_starlette emits CRLF
    // ("\r\n\r\n"); split on the optional-CRLF form so both "\n\n" and
    // "\r\n\r\n" separate frames (a bare split("\n\n") silently swallows
    // every token because "\r\n\r\n" never contains "\n\n").
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const ev = parseFrame(frame);
      if (ev) handlers.onEvent(ev);
    }
  }
}

export function parseFrame(frame: string): StreamEvent | null {
  let event = "message";
  let data = "";
  for (const line of frame.split(/\r?\n/)) {
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

// Live SSE subscription for cycle progress (worker tokens/tools/gate/merge).
// `onEvent` receives `cycle_event` / `cycle_result` / `ping` frames; the
// returned AbortController lets the caller stop the infinite stream.
export function streamCycleEvents(handlers: StreamHandlers): AbortController {
  const controller = new AbortController();
  (async () => {
    try {
      const resp = await fetch("/api/cycles/stream", { signal: controller.signal });
      await pumpStream(resp, handlers);
    } catch (err) {
      // AbortError is expected when the caller stops the stream; only surface
      // real failures.
      if ((err as Error).name !== "AbortError") {
        handlers.onError?.(err as Error);
      }
    }
  })();
  return controller;
}
