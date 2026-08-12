// Shared types mirroring the FastAPI JSON shapes (services.py).

export type Role = "user" | "assistant" | "tool";

export interface ToolCall {
  id?: string;
  name?: string;
  arguments?: Record<string, unknown>;
  // legacy field — older rows may store args flat under tool_calls entries
  [key: string]: unknown;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  parent_id: string | null;
  is_active: boolean;
  is_deleted?: boolean;
  sibling_order: number;
  created_at?: string;
}

export interface ChatSummary {
  id: string;
  title: string;
  frontend: string;
  started_at: string;
  ended_at: string | null;
  message_count: number;
  last_message_at: string | null;
}

export interface ChatDetail extends ChatSummary {
  messages: Message[];
  tree: MessageTree;
  active_leaf_id: string | null;
  memory: { name: string; size: number }[];
}

export interface MessageTree {
  root_id: string | null;
  nodes: Record<string, Message>;
  // children of a parent_id; root-level messages hang under the key "null"
  children: Record<string, Message[] | undefined>;
}

export interface Profile {
  name: string;
  base_url: string;
  model: string;
}

export type CycleStatus = "running" | "merged" | "rejected" | "stuck" | string;

export interface CycleStep {
  step: number;
  status: string;
  commit_sha: string | null;
  summary: string | null;
}

export interface CycleSummary {
  id: string;
  objective: string;
  branch: string | null;
  status: CycleStatus;
  ai_verdict: string | null;
  human_verdict: string | null;
  started_at: string | null;
  ended_at: string | null;
  steps: CycleStep[];
  // Parallel cycles fill one entry per planner slice; single-worker cycles
  // (and older DBs) leave this empty and the UI falls back to the step list.
  workers: CycleWorker[];
  human_review_id: string | null;
}

export interface CycleWorker {
  id: string;
  worker_index: number;
  title: string;
  detail: string;
  status: string;
  started_at: string | null;
  ended_at: string | null;
}

export interface CycleEvent {
  id: number;
  kind: string;
  message: string;
  payload: Record<string, unknown>;
  seq: number;
}

export interface CycleReview {
  id: string;
  kind: string;
  verdict: string;
  comments: string;
  resolved_at: string | null;
}

export interface CycleDetail extends CycleSummary {
  events: CycleEvent[];
  reviews: CycleReview[];
}

export interface MemoryFile {
  name: string;
  size: number;
}

export interface UsageTotals {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cache_read_tokens: number;
  cache_read_pct: number;
  calls: number;
}

export interface UsageEvent {
  id: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cache_read_tokens: number;
  cache_read_pct: number;
  created_at: string;
  session_id: string | null;
  cycle_id: string | null;
}

export interface UsageAggregate {
  totals: UsageTotals;
  events: UsageEvent[];
}

// SSE events emitted by the chat / regenerate endpoints, plus the cycle live
// stream (event names `cycle_event`/`cycle_result`/`ping` come from web.py).
export type StreamEvent =
  | { event: "token"; data: { text: string } }
  | { event: "tool"; data: { name: string; args: Record<string, unknown> } }
  | { event: "tool_result"; data: { name: string; snippet: string } }
  | { event: "usage"; data: Record<string, number> }
  | {
      event: "done";
      data: {
        answer: string;
        usage?: Record<string, number>;
        chat_id: string;
        user_message_id?: string | null;
        assistant_message_id?: string | null;
      };
    }
  | { event: "error"; data: { message: string } }
  // ---- cycle live stream (web.py `/api/cycles/stream`) ----
  | {
      event: "cycle_event";
      data: {
        cycle_id: string;
        kind: string;
        message: string;
        payload: Record<string, unknown>;
      };
    }
  | {
      event: "cycle_result";
      data: { cycle_id: string; status: string; branch: string; steps: number };
    }
  | { event: "ping"; data: Record<string, unknown> };
