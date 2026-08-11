import { Disclosure } from "@headlessui/react";

export interface ToolEntry {
  name: string;
  args: Record<string, unknown>;
  snippet?: string;
}

function formatArgs(args: Record<string, unknown>): string {
  try {
    return JSON.stringify(args);
  } catch {
    return String(args);
  }
}

export function ToolCallBlock({ tool }: { tool: ToolEntry }) {
  const args = formatArgs(tool.args);
  const preview = args.length > 80 ? args.slice(0, 80) + "…" : args;
  return (
    <Disclosure as="div" className="my-1.5 rounded-lg border border-edge bg-panel2/60">
      <Disclosure.Button className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs text-zinc-400 hover:text-zinc-200">
        <span className="text-accent">↹</span>
        <span className="font-mono font-medium text-zinc-300">{tool.name}</span>
        <span className="truncate font-mono text-[11px] text-zinc-500">{preview}</span>
        <span className="ml-auto text-zinc-600">▾</span>
      </Disclosure.Button>
      <Disclosure.Panel className="border-t border-edge px-3 py-2 text-xs">
        {args && (
          <div className="mb-2">
            <div className="mb-0.5 text-[10px] uppercase tracking-wide text-zinc-600">args</div>
            <pre className="overflow-x-auto font-mono text-zinc-300">{args}</pre>
          </div>
        )}
        {tool.snippet !== undefined && (
          <div>
            <div className="mb-0.5 text-[10px] uppercase tracking-wide text-zinc-600">result</div>
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words font-mono text-zinc-400">
              {tool.snippet}
            </pre>
          </div>
        )}
      </Disclosure.Panel>
    </Disclosure>
  );
}
