import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCyclesStore } from "./cyclesStore";
import type { StreamEvent } from "@/state/types";

// Drive the SSE subscription synchronously from a list of canned events so we
// can assert how the store folds worker events into per-worker cards.
function mockStream(events: StreamEvent[]) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const ev of events) {
        const data = ev.event === "ping" ? "{}" : JSON.stringify(ev.data);
        controller.enqueue(encoder.encode(`event: ${ev.event}\r\ndata: ${data}\r\n\r\n`));
      }
      controller.close();
    },
  });
  const response = new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
  globalThis.fetch = vi.fn().mockResolvedValue(response) as never;
}

describe("cyclesStore live worker stream", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    useCyclesStore.setState({
      cycles: [],
      detail: null,
      loading: false,
      error: null,
      lastStartedId: null,
      liveWorkers: {},
      liveCycleId: null,
      liveController: null,
    });
  });

  afterEach(() => {
    useCyclesStore.getState().stopLive();
    globalThis.fetch = originalFetch;
  });

  it("folds per-worker events into separate cards keyed by worker_id", async () => {
    mockStream([
      {
        event: "cycle_event",
        data: {
          cycle_id: "c1",
          kind: "worker_start",
          message: "worker 0 started",
          payload: { worker_id: "w0", worker_index: 0, title: "task-A" },
        },
      },
      {
        event: "cycle_event",
        data: {
          cycle_id: "c1",
          kind: "agent_token",
          message: "Hi",
          payload: { worker_id: "w0", token: "Hi" },
        },
      },
      {
        event: "cycle_event",
        data: {
          cycle_id: "c1",
          kind: "worker_start",
          message: "worker 1 started",
          payload: { worker_id: "w1", worker_index: 1, title: "task-B" },
        },
      },
      {
        event: "cycle_event",
        data: {
          cycle_id: "c1",
          kind: "agent_tool",
          message: "self_write",
          payload: { worker_id: "w1", tool: "self_write", args: { path: "x.md" } },
        },
      },
      {
        event: "cycle_event",
        data: {
          cycle_id: "c1",
          kind: "worker_done",
          message: "done",
          payload: { worker_id: "w0", answer: "all good" },
        },
      },
    ]);

    useCyclesStore.getState().startLive("c1");
    // Let the fetch + pumpStream microtasks run to completion.
    await new Promise((r) => setTimeout(r, 50));

    const { liveWorkers } = useCyclesStore.getState();
    const w0 = liveWorkers.w0;
    const w1 = liveWorkers.w1;
    expect(w0).toBeDefined();
    expect(w1).toBeDefined();
    expect(w0.title).toBe("task-A");
    expect(w0.status).toBe("done");
    expect(w0.content).toBe("Hi"); // token accumulated
    expect(w1.title).toBe("task-B");
    expect(w1.status).toBe("running");
    expect(w1.tools).toHaveLength(1);
    expect(w1.tools[0].name).toBe("self_write");
  });

  it("ignores events for other cycles when a filter is set", async () => {
    mockStream([
      {
        event: "cycle_event",
        data: {
          cycle_id: "other-cycle",
          kind: "worker_start",
          message: "x",
          payload: { worker_id: "wx", worker_index: 0, title: "other" },
        },
      },
    ]);

    useCyclesStore.getState().startLive("c1");
    await new Promise((r) => setTimeout(r, 50));

    const { liveWorkers } = useCyclesStore.getState();
    expect(liveWorkers.wx).toBeUndefined();
  });

  it("startCycle posts objective and auto-approve to /api/improve", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: "started", cycle_id: "c-new" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    globalThis.fetch = fetchMock as never;
    // listChats fallback inside loadCycles — stub it too.
    const cyclesListMock = vi.fn().mockResolvedValue([]);
    // loadCycles uses the same global fetch, so a single mock that answers
    // based on URL is enough.
    fetchMock.mockImplementation(async (url: string) => {
      if (String(url).includes("/api/improve")) {
        return new Response(JSON.stringify({ status: "started", cycle_id: "c-new" }), {
          status: 200, headers: { "content-type": "application/json" },
        });
      }
      return new Response("[]", {
        status: 200, headers: { "content-type": "application/json" },
      });
    });

    const id = await useCyclesStore.getState().startCycle("improve the thing", true);

    expect(id).toBe("c-new");
    expect(useCyclesStore.getState().lastStartedId).toBe("c-new");
    const improveCall = fetchMock.mock.calls.find((c) => String(c[0]).includes("/api/improve"));
    expect(improveCall).toBeDefined();
    const init = improveCall![1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ objective: "improve the thing", auto_approve: true });

    globalThis.fetch = cyclesListMock as never;
  });
});
