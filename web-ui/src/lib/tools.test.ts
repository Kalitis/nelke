import { describe, expect, it } from "vitest";
import { toolEntriesFromMessage } from "./tools";
import type { Message, MessageTree } from "@/state/types";

function treeWith(nodes: Record<string, Message>): MessageTree {
  const children: MessageTree["children"] = {};
  for (const n of Object.values(nodes)) {
    const key = n.parent_id ?? "null";
    (children[key] ??= []).push(n);
  }
  return { root_id: nodes.a?.id ?? null, nodes, children };
}

function toolResult(id: string, parentId: string, content: string): Message {
  return {
    id,
    role: "tool",
    content,
    parent_id: parentId,
    is_active: true,
    is_deleted: false,
    sibling_order: 0,
  };
}

function assistant(
  id: string,
  toolCalls: Message["tool_calls"],
  parentId: string | null = null,
): Message {
  return {
    id,
    role: "assistant",
    content: "",
    tool_calls: toolCalls,
    parent_id: parentId,
    is_active: true,
    is_deleted: false,
    sibling_order: 0,
  };
}

describe("toolEntriesFromMessage", () => {
  it("builds collapsed tool entries from a persisted tool-calling assistant message", () => {
    const a = assistant("a", [
      {
        id: "call-1",
        type: "function",
        function: { name: "web_fetch", arguments: '{"url": "https://example.com"}' },
      },
    ]);
    const t1 = toolResult("t1", "a", "200 OK — <html>…");
    t1.tool_call_id = "call-1";
    const tree = treeWith({ a, t1 });

    const entries = toolEntriesFromMessage(a, tree);
    expect(entries).toEqual([
      { name: "web_fetch", args: { url: "https://example.com" }, snippet: "200 OK — <html>…" },
    ]);
  });

  it("returns an empty list when the assistant made no tool calls", () => {
    const a = assistant("a", undefined);
    const tree = treeWith({ a });
    expect(toolEntriesFromMessage(a, tree)).toEqual([]);
  });

  it("skips malformed tool_calls (no function.name)", () => {
    const a = assistant("a", [
      { id: "call-1", type: "function" },
      { id: "call-2", type: "function", function: { name: "grep", arguments: "{}" } },
    ] as Message["tool_calls"]);
    const tree = treeWith({ a });
    const entries = toolEntriesFromMessage(a, tree);
    expect(entries).toHaveLength(1);
    expect(entries[0].name).toBe("grep");
  });

  it("leaves the snippet undefined when the result message is missing", () => {
    const a = assistant("a", [
      { id: "call-1", type: "function", function: { name: "bash", arguments: "{}" } },
    ]);
    const tree = treeWith({ a });
    const entries = toolEntriesFromMessage(a, tree);
    expect(entries).toEqual([{ name: "bash", args: {}, snippet: undefined }]);
  });

  it("ignores deleted tool results and pairs results by tool_call_id", () => {
    const a = assistant("a", [
      { id: "call-1", type: "function", function: { name: "recall", arguments: '{"query":"q"}' } },
      { id: "call-2", type: "function", function: { name: "memory_show", arguments: "{}" } },
    ]);
    const deleted = toolResult("t-deleted", "a", "deleted result");
    deleted.is_deleted = true;
    deleted.tool_call_id = "call-1";
    const t2 = toolResult("t2", "a", "memory file body");
    t2.tool_call_id = "call-2";
    const tree = treeWith({ a, "t-deleted": deleted, t2 });

    const entries = toolEntriesFromMessage(a, tree);
    expect(entries).toHaveLength(2);
    expect(entries[0].name).toBe("recall");
    expect(entries[0].snippet).toBeUndefined();
    expect(entries[1].name).toBe("memory_show");
    expect(entries[1].snippet).toBe("memory file body");
  });
});
