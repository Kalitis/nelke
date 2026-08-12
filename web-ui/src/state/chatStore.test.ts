import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useChatStore, visibleTranscript } from "./chatStore";
import type { ChatDetail } from "@/state/types";

// Build a chat detail with a minimal tree: one assistant message (no parent).
function chatFixture(): ChatDetail {
  return {
    id: "c1",
    title: "test chat",
    frontend: "web",
    started_at: "2026-01-01T00:00:00",
    ended_at: null,
    message_count: 1,
    last_message_at: null,
    messages: [
      {
        id: "a1",
        role: "assistant",
        content: "hi",
        parent_id: null,
        is_active: true,
        is_deleted: false,
        sibling_order: 0,
      },
    ],
    tree: {
      root_id: "a1",
      nodes: {
        a1: {
          id: "a1",
          role: "assistant",
          content: "hi",
          parent_id: null,
          is_active: true,
          is_deleted: false,
          sibling_order: 0,
        },
      },
      children: { null: [{}] as never },
    },
    active_leaf_id: "a1",
    memory: [],
  };
}

// A streaming Response that never resolves the body (simulates an in-flight
// turn): the reader's first `read()` hangs until the test finishes. This lets
// us assert the optimistic state during streaming.
function hangingResponse(): Response {
  const stream = new ReadableStream<Uint8Array>({
    start() {
      // Intentionally never enqueue or close.
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

describe("chatStore optimistic user message", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    useChatStore.setState({
      chats: [],
      activeChatId: "c1",
      chat: chatFixture(),
      profiles: [],
      profile: null,
      streaming: false,
      streamBuffer: { content: "", tools: [] },
      optimisticMessageId: null,
      error: null,
    });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("inserts the user message into the tree immediately (before the assistant replies)", async () => {
    // fetch resolves once with a hanging stream — never reaches `done`.
    globalThis.fetch = vi.fn().mockResolvedValue(hangingResponse()) as never;

    // Kick off sendMessage; do not await the full run (it never ends).
    void useChatStore.getState().sendMessage("hello world");

    // Let the fetch promise + pumpStream kick off on the microtask queue.
    await Promise.resolve();
    await Promise.resolve();

    const { chat, optimisticMessageId } = useChatStore.getState();
    expect(chat).not.toBeNull();
    expect(optimisticMessageId).not.toBeNull();

    const transcript = visibleTranscript(chat);
    const ids = transcript.map((m) => m.id);
    // The optimistic user message appears as a new leaf under the assistant.
    expect(ids).toContain(optimisticMessageId);
    const leaf = transcript[transcript.length - 1];
    expect(leaf.role).toBe("user");
    expect(leaf.content).toBe("hello world");
    expect(leaf.parent_id).toBe("a1");
  });

  it("rolls back the optimistic message when the stream errors out", async () => {
    // fetch rejects immediately — runStream catches and rolls back.
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("network down")) as never;

    await useChatStore.getState().sendMessage("oops");

    const { chat, optimisticMessageId, error } = useChatStore.getState();
    expect(optimisticMessageId).toBeNull();
    expect(error).toContain("network down");
    expect(chat).not.toBeNull();
    // The tree is back to the original single-node shape.
    const transcript = visibleTranscript(chat);
    expect(transcript.map((m) => m.id)).toEqual(["a1"]);
  });
});
