import { create } from "zustand";
import { api, streamCycleEvents } from "@/api/client";
import type { CycleDetail, CycleSummary, CycleWorker } from "@/state/types";

interface LiveTool {
  name: string;
  args: Record<string, unknown>;
  snippet?: string;
}

export interface SubagentCard {
  id: string;
  worker_index: number;
  title: string;
  status: string;
  content: string;
  tools: LiveTool[];
  lastEvent: string;
  lastEventKind: string;
}

interface CyclesState {
  cycles: CycleSummary[];
  detail: CycleDetail | null;
  loading: boolean;
  error: string | null;
  // id of the most recently started cycle (when known) so the UI can navigate
  // to its detail page right after `startCycle` returns.
  lastStartedId: string | null;
  // live worker cards keyed by worker_id; populated from the cycle SSE stream.
  liveWorkers: Record<string, SubagentCard>;
  liveCycleId: string | null;
  // AbortController for the live SSE subscription; stopped on `stopLive`.
  liveController: AbortController | null;

  loadCycles: () => Promise<void>;
  loadDetail: (id: string) => Promise<void>;
  clearDetail: () => void;
  startCycle: (objective: string, autoApprove: boolean) => Promise<string | null>;
  // Subscribe to the cycle live stream; events are routed to `liveWorkers` by
  // the `worker_id` payload field. Pass a cycle_id to filter to a single cycle
  // (events for other cycles are ignored).
  startLive: (cycleId?: string | null) => void;
  stopLive: () => void;
}

export const useCyclesStore = create<CyclesState>((set, get) => ({
  cycles: [],
  detail: null,
  loading: false,
  error: null,
  lastStartedId: null,
  liveWorkers: {},
  liveCycleId: null,
  liveController: null,

  loadCycles: async () => {
    try {
      const cycles = await api.cyclesList();
      set({ cycles, error: null });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  loadDetail: async (id) => {
    set({ loading: true, error: null });
    try {
      const detail = await api.cycleDetail(id);
      set({ detail, loading: false });
    } catch (err) {
      set({ loading: false, error: String(err), detail: null });
    }
  },

  clearDetail: () => set({ detail: null, loading: false }),

  startCycle: async (objective, autoApprove) => {
    try {
      const result = await api.startCycle(objective, autoApprove);
      if (result.cycle_id) set({ lastStartedId: result.cycle_id });
      // Refresh the list so the new cycle shows up even if cycle_id was not
      // returned (engine slow to emit `cycle_start`).
      await get().loadCycles();
      return result.cycle_id ?? null;
    } catch (err) {
      set({ error: String(err) });
      return null;
    }
  },

  startLive: (cycleId) => {
    // If already subscribed, no-op (the caller may re-issue on re-render).
    if (get().liveController) return;
    const filterCycleId = cycleId ?? null;
    set({ liveCycleId: filterCycleId });
    const controller = streamCycleEvents({
      onEvent: (ev) => {
        if (ev.event === "ping") return;
        if (ev.event === "cycle_result") {
          // Cycle finished; refresh the canonical detail so final worker
          // statuses / merge result are reflected.
          const id = get().liveCycleId ?? ev.data.cycle_id;
          void get().loadDetail(id);
          return;
        }
        if (ev.event !== "cycle_event") return;
        const { cycle_id, kind, message, payload } = ev.data;
        // Ignore events for other cycles when a filter is set.
        if (filterCycleId && cycle_id !== filterCycleId) return;
        const workerId = (payload.worker_id as string | undefined) ?? null;
        // Some events (planned, gate, commit) are cycle-wide and not worker-
        // specific; they still update the detail on the next poll.
        if (!workerId) return;
        applyWorkerEvent(set, get, workerId, kind, message, payload);
      },
      onError: (err) => set({ error: String(err) }),
    });
    set({ liveController: controller });
  },

  stopLive: () => {
    const controller = get().liveController;
    if (controller) controller.abort();
    set({ liveController: null, liveCycleId: null, liveWorkers: {} });
  },
}));

/**
 * Fold one cycle SSE event into the worker card it belongs to. Tokens
 * accumulate into `content`; tool calls become `LiveTool` entries; other
 * kinds bump the card status and lastEvent fields.
 */
function applyWorkerEvent(
  set: (partial: Partial<CyclesState>) => void,
  get: () => CyclesState,
  workerId: string,
  kind: string,
  message: string,
  payload: Record<string, unknown>,
): void {
  const current = get().liveWorkers[workerId];
  // Seed a card from the detail's `workers` list if we have not seen this
  // worker yet (the cycle_events stream usually starts mid-flight).
  const seed: SubagentCard = current ?? seedFromDetail(get, workerId) ?? {
    id: workerId,
    worker_index: payload.worker_index ?? 0,
    title: payload.title ?? workerId,
    status: "running",
    content: "",
    tools: [],
    lastEvent: message,
    lastEventKind: kind,
  };

  let next: SubagentCard;
  switch (kind) {
    case "agent_token":
      next = { ...seed, content: seed.content + (payload.token ?? ""), lastEvent: message, lastEventKind: kind };
      break;
    case "agent_tool":
      next = {
        ...seed,
        tools: [
          ...seed.tools,
          {
            name: (payload.tool as string | undefined) ?? "?",
            args: (payload.args as Record<string, unknown> | undefined) ?? {},
          },
        ],
        lastEvent: message,
        lastEventKind: kind,
      };
      break;
    case "agent_tool_result":
      next = {
        ...seed,
        tools: seed.tools.map((t, i) =>
          i === seed.tools.length - 1 && !t.snippet
            ? { ...t, snippet: (payload.snippet as string | undefined) ?? "" }
            : t,
        ),
        lastEvent: message,
        lastEventKind: kind,
      };
      break;
    case "worker_start":
      next = { ...seed, status: "running", title: (payload.title as string) ?? seed.title, lastEvent: message, lastEventKind: kind };
      break;
    case "worker_done":
      next = { ...seed, status: "done", lastEvent: message, lastEventKind: kind };
      break;
    case "worker_error":
      next = { ...seed, status: "error", lastEvent: message, lastEventKind: kind };
      break;
    default:
      next = { ...seed, lastEvent: message, lastEventKind: kind };
  }
  set({ liveWorkers: { ...get().liveWorkers, [workerId]: next } });
}

/** Look up a worker's static metadata (title/index) from the loaded detail. */
function seedFromDetail(get: () => CyclesState, workerId: string): SubagentCard | null {
  const detail = get().detail;
  if (!detail) return null;
  const w: CycleWorker | undefined = detail.workers.find((x) => x.id === workerId);
  if (!w) return null;
  return {
    id: w.id,
    worker_index: w.worker_index,
    title: w.title,
    status: w.status,
    content: "",
    tools: [],
    lastEvent: "",
    lastEventKind: "",
  };
}
