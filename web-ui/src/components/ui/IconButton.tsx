import { type ButtonHTMLAttributes, forwardRef } from "react";

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  title: string;
}

/** Small ghost icon button used for message hover-actions. */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton({ title, className = "", children, ...rest }, ref) {
    return (
      <button
        ref={ref}
        type="button"
        title={title}
        aria-label={title}
        className={
          "rounded-md p-1.5 text-zinc-400 transition-colors " +
          "hover:bg-panel2 hover:text-zinc-100 disabled:cursor-not-allowed disabled:opacity-40 " +
          className
        }
        {...rest}
      >
        {children}
      </button>
    );
  },
);
