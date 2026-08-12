import type { Message, MessageTree } from "@/state/types";
import type { ToolEntry } from "@/components/chat/ToolCallBlock";
import { childrenOf } from "@/lib/tree";

interface PersistedToolCall {
  id?: string;
  function: { name: string; arguments: string };
}

function parseArgs(raw: string | undefined): Record<string, unknown> {
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

/**
 * Build collapsible ToolEntry blocks for a persisted assistant message that
 * called tools. The tool arguments come from the message's ``tool_calls`` and
 * the result snippet from the matching ``tool`` child message(s).
 *
 * Tool-result rows are separate ``tool`` messages in the tree; without this
 * folding they render as full assistant bubbles and dump raw tool output into
 * the conversation. With it, tool work stays in the collapsed blocks shown
 * live during streaming — same shape, now persisted too.
 */
export function toolEntriesFromMessage(node: Message, tree: MessageTree): ToolEntry[] {
  // tool_calls arrives as parsed JSON (trusted server data); validate the
  // shape before reading `function`.
  const calls = ((node.tool_calls as unknown as PersistedToolCall[] | undefined) ?? []).filter(
    (tc) =>
      !!tc &&
      typeof tc.function === "object" &&
      tc.function !== null &&
      typeof tc.function.name === "string",
  );
  if (calls.length === 0) return [];
  const results = childrenOf(tree, node.id).filter(
    (c) => c.role === "tool" && !c.is_deleted && !!c.content,
  );
  return calls.map((tc, i) => {
    const res = tc.id
      ? results.find((r) => r.tool_call_id === tc.id)
      : results[i];
    return {
      name: tc.function.name,
      args: parseArgs(tc.function.arguments),
      snippet: res ? res.content.slice(0, 200) : undefined,
    };
  });
}
