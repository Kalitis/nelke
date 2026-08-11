import { useEffect, useRef, useState } from "react";

interface EditComposerProps {
  initial: string;
  onSave: (text: string) => void;
  onCancel: () => void;
}

/** Inline textarea for editing a user message before regeneration. */
export function EditComposer({ initial, onSave, onCancel }: EditComposerProps) {
  const [value, setValue] = useState(initial);
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    ref.current?.focus();
    ref.current?.setSelectionRange(initial.length, initial.length);
  }, [initial]);

  return (
    <div className="rounded-lg border border-edge bg-panel2 p-2">
      <textarea
        ref={ref}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (value.trim()) onSave(value.trim());
          } else if (e.key === "Escape") {
            e.preventDefault();
            onCancel();
          }
        }}
        rows={3}
        className="w-full resize-y rounded bg-canvas px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none"
      />
      <div className="mt-2 flex justify-end gap-2">
        <button
          type="button"
          onClick={onCancel}
          className="rounded-lg px-3 py-1.5 text-xs text-zinc-400 hover:bg-panel"
        >
          Cancel
        </button>
        <button
          type="button"
          disabled={!value.trim()}
          onClick={() => value.trim() && onSave(value.trim())}
          className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-canvas disabled:opacity-40"
        >
          Save &amp; submit
        </button>
      </div>
    </div>
  );
}
