import { IconButton } from "@/components/ui/IconButton";

interface MessageActionsProps {
  role: "user" | "assistant";
  content: string;
  onEdit: () => void;
  onRegenerate: () => void;
  onDelete: () => void;
}

/** Hover-action toolbar shown at the bottom of a message bubble. */
export function MessageActions({
  role,
  content,
  onEdit,
  onRegenerate,
  onDelete,
}: MessageActionsProps) {
  return (
    <div className="mt-1 flex items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
      <IconButton
        title="Copy"
        onClick={() => void navigator.clipboard.writeText(content)}
      >
        ⧉
      </IconButton>
      {role === "user" && (
        <IconButton title="Edit" onClick={onEdit}>
          ✎
        </IconButton>
      )}
      {role === "assistant" && (
        <IconButton title="Regenerate" onClick={onRegenerate}>
          ↻
        </IconButton>
      )}
      <IconButton title="Delete" onClick={onDelete}>
        🗑
      </IconButton>
    </div>
  );
}
