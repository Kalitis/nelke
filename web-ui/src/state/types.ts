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

export interface UsageTotals {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  calls: number;
}

// SSE events emitted by the chat / regenerate endpoints.
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
  | { event: "error"; data: { message: string } };
