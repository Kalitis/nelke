import { afterEach, describe, expect, it, vi } from "vitest";
import { api, parseFrame, pumpStream } from "./client";
import type { StreamEvent } from "@/state/types";

// SSE frames as emitted by sse_starlette use CRLF ("\r\n") line endings and a
// blank line ("\r\n\r\n") between frames. The streaming client must split on
// both CRLF and bare-LF separators — a naive split("\n\n") swallows every
// frame because "\r\n\r\n" never contains "\n\n".

describe("parseFrame", () => {
  it("parses a token frame with LF line endings", () => {
    const ev = parseFrame("event: token\ndata: {\"text\": \"Hi\"}");
    expect(ev).toEqual<StreamEvent>({ event: "token", data: { text: "Hi" } });
  });

  it("parses a token frame with CRLF line endings (sse_starlette shape)", () => {
    const ev = parseFrame("event: token\r\ndata: {\"text\": \"Hi\"}\r\n");
    expect(ev).toEqual<StreamEvent>({ event: "token", data: { text: "Hi" } });
  });

  it("falls back to a token event when data is not JSON", () => {
    const ev = parseFrame("data: plain text");
    expect(ev).toEqual<StreamEvent>({ event: "token", data: { text: "plain text" } });
  });

  it("returns null for an empty frame", () => {
    expect(parseFrame("")).toBeNull();
    expect(parseFrame("event: token")).toBeNull();
  });

  it("parses a done frame carrying the assistant message ids", () => {
    const frame = [
      'event: done',
      'data: {"answer":"ok","usage":{"total_tokens":1},"chat_id":"c1","user_message_id":"u1","assistant_message_id":"a1"}',
    ].join("\n");
    const ev = parseFrame(frame);
    expect(ev?.event).toBe("done");
    expect(ev?.data).toMatchObject({ chat_id: "c1", assistant_message_id: "a1" });
  });
});

// Build a Response whose body yields the given chunks sequentially, mimicking
// a streamed SSE response (fetch ReadableStream). CRLF separators are included
// verbatim to reproduce the production byte shape.
function streamedResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
  // fetch Response is available globally in jsdom via undici.
  return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
}

describe("pumpStream", () => {
  it("emits events split on CRLF blank-line separators", async () => {
    const body =
      'event: token\r\ndata: {"text": "Hel"}\r\n\r\n' +
      'event: token\r\ndata: {"text": "lo"}\r\n\r\n' +
      'event: usage\r\ndata: {"total_tokens": 2}\r\n\r\n' +
      'event: done\r\ndata: {"answer": "Hello", "chat_id": "c1"}\r\n\r\n';
    const events: StreamEvent[] = [];
    await pumpStream(streamedResponse([body]), { onEvent: (ev) => events.push(ev) });
    expect(events.map((e) => e.event)).toEqual(["token", "token", "usage", "done"]);
    expect(events[0]).toEqual<StreamEvent>({ event: "token", data: { text: "Hel" } });
    expect(events[3].data).toMatchObject({ answer: "Hello", chat_id: "c1" });
  });

  it("emits events split on bare-LF separators too", async () => {
    const body =
      'event: token\ndata: {"text": "A"}\n\n' +
      'event: token\ndata: {"text": "B"}\n\n';
    const events: StreamEvent[] = [];
    await pumpStream(streamedResponse([body]), { onEvent: (ev) => events.push(ev) });
    expect(events.map((e) => e.event)).toEqual(["token", "token"]);
  });

  it("flushes a trailing frame without a closing separator on stream end", async () => {
    // sse_starlette may close the response immediately after the final frame
    // without a trailing "\r\n\r\n". The terminal event (e.g. `done`) must
    // still be delivered — otherwise the canonical chat tree never reloads.
    const events: StreamEvent[] = [];
    const handlers = { onEvent: (ev: StreamEvent) => events.push(ev) };
    await pumpStream(
      streamedResponse([
        'event: token\r\ndata: {"text": "X"}\r\n\r\n',
        'event: done\r\ndata: {"answer": "X", "chat_id": "c1"}',
      ]),
      handlers,
    );
    expect(events.map((e) => e.event)).toEqual(["token", "done"]);
    expect(events[1].data).toMatchObject({ answer: "X", chat_id: "c1" });
  });

  it("emits a partial frame only once when no separator arrives and stream ends", async () => {
    // A single partial frame (no separator) still resolves to one event on
    // stream close — it is not lost, and it is not duplicated.
    const events: StreamEvent[] = [];
    const handlers = { onEvent: (ev: StreamEvent) => events.push(ev) };
    await pumpStream(streamedResponse(['event: token\r\ndata: {"text": "X"}']), handlers);
    expect(events).toHaveLength(1);
    expect(events[0]).toEqual<StreamEvent>({ event: "token", data: { text: "X" } });
  });

  it("emits frames incrementally as chunks arrive across reads", async () => {
    const events: StreamEvent[] = [];
    const handlers = { onEvent: (ev: StreamEvent) => events.push(ev) };
    await pumpStream(
      streamedResponse([
        'event: token\r\ndata: {"text": "1"}\r\n\r\n',
        'event: token\r\ndata: {"text": "2"}\r\n\r\n',
      ]),
      handlers,
    );
    expect(events.map((e) => e.data)).toEqual([
      { text: "1" },
      { text: "2" },
    ]);
  });

  it("surfaces error events", async () => {
    const events: StreamEvent[] = [];
    await pumpStream(
      streamedResponse(['event: error\r\ndata: {"message": "boom"}\r\n\r\n']),
      { onEvent: (ev) => events.push(ev) },
    );
    expect(events).toHaveLength(1);
    expect(events[0]).toEqual<StreamEvent>({ event: "error", data: { message: "boom" } });
  });

  it("throws on a non-2xx response", async () => {
    const bad = new Response("nope", { status: 500 });
    await expect(pumpStream(bad, { onEvent: () => {} })).rejects.toThrow("HTTP 500");
  });
});

describe("api.usage", () => {
  let originalFetch: typeof globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("requests the persisted usage for a session and returns its totals", async () => {
    const totals = {
      prompt_tokens: 100,
      completion_tokens: 50,
      total_tokens: 150,
      cache_read_tokens: 40,
      cache_read_pct: 26,
      calls: 3,
    };
    const payload = { totals, events: [] };
    let requestedUrl = "";
    originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockImplementation((url: RequestInfo | URL) => {
      requestedUrl = String(url);
      return Promise.resolve(
        new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
    }) as never;

    const result = await api.usage("chat-123");
    expect(requestedUrl).toBe("/api/usage?session_id=chat-123");
    expect(result.totals.total_tokens).toBe(150);
  });
});

describe("api.projects", () => {
  let originalFetch: typeof globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  function mockFetch(routes: Record<string, (body: any) => unknown>) {
    originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? "GET").toUpperCase();
      const body = init?.body ? JSON.parse(String(init.body)) : {};
      for (const key of Object.keys(routes)) {
        const [m, path] = key.split(" ");
        if (method === m && url === path) {
          return Promise.resolve(
            new Response(JSON.stringify(routes[key](body)), {
              status: 200,
              headers: { "content-type": "application/json" },
            }),
          ) as never;
        }
      }
      return Promise.resolve(new Response("not found", { status: 404 })) as never;
    }) as never;
  }

  it("lists projects via GET /api/projects", async () => {
    const projects = [{ id: "p1", name: "Nelke", stage: "idea", chat_count: 0 }];
    mockFetch({ "GET /api/projects": () => projects });
    const result = await api.projectsList();
    expect(result).toHaveLength(1);
    expect(result[0].name).toBe("Nelke");
  });

  it("creates a project via POST with name/description/stage", async () => {
    let captured: any = null;
    originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      captured = { url: String(input), method: init?.method, body: JSON.parse(String(init?.body ?? "{}")) };
      return Promise.resolve(
        new Response(JSON.stringify({ id: "p1", name: "Nelke" }), {
          status: 200, headers: { "content-type": "application/json" },
        }),
      ) as never;
    }) as never;

    const result = await api.createProject("Nelke", { description: "agent", stage: "idea" });
    expect(captured.url).toBe("/api/projects");
    expect(captured.method).toBe("POST");
    expect(captured.body).toEqual({ name: "Nelke", description: "agent", stage: "idea" });
    expect(result.id).toBe("p1");
  });

  it("links a chat via POST /api/projects/:id/chats/:chatId", async () => {
    let requestedUrl = "";
    mockFetch({
      "POST /api/projects/p1/chats/c1": () => {
        requestedUrl = "/api/projects/p1/chats/c1";
        return { ok: true };
      },
    });
    const result = await api.linkChat("p1", "c1");
    expect(requestedUrl).toBe("/api/projects/p1/chats/c1");
    expect(result.ok).toBe(true);
  });
});
