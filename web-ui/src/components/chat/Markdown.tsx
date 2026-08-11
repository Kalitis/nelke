import { memo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

function CodeBlockCopyButton({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        void navigator.clipboard.writeText(code).then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        });
      }}
      className="absolute right-2 top-2 rounded border border-edge bg-panel/80 px-2 py-0.5 text-[11px] text-zinc-400 opacity-0 transition-opacity hover:text-zinc-100 group-hover:opacity-100"
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export const Markdown = memo(function Markdown({ content }: { content: string }) {
  return (
    <div className="prose-nelke text-zinc-100">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={{
          pre({ children }) {
            // Extract raw text for the copy button.
            let code = "";
            const child: any = Array.isArray(children) ? children[0] : children;
            if (child?.props?.children) {
              code = String(child.props.children);
            }
            return (
              <div className="group relative">
                <CodeBlockCopyButton code={code} />
                <pre>{children}</pre>
              </div>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});
