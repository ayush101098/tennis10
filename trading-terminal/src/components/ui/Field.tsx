"use client";

import { forwardRef, useId } from "react";
import type { InputHTMLAttributes, SelectHTMLAttributes, ReactNode } from "react";

/**
 * Form controls — see DESIGN.md §7.
 *
 * One height, one border, one focus treatment. Labels are required rather than
 * optional: a placeholder is not a label, because it disappears exactly when
 * the user needs to check what they are filling in.
 *
 * Errors are wired through `aria-describedby` so a screen reader reads the
 * message with the field rather than leaving it stranded in the DOM.
 */

const CONTROL =
  "w-full bg-bg border rounded-sm text-content placeholder:text-content-faint " +
  "transition-colors duration-fast ease-standard " +
  "disabled:opacity-50 disabled:cursor-not-allowed";

const sizeCls = (size: "sm" | "md") => (size === "sm" ? "h-7 px-2 text-xs" : "h-9 px-3 text-sm");

interface WrapProps {
  label: string;
  /** Hide the label visually but keep it for assistive tech. */
  hideLabel?: boolean;
  hint?: string;
  error?: string;
  required?: boolean;
  children: (ids: { id: string; describedBy?: string; invalid: boolean }) => ReactNode;
}

/** Label + control + hint/error, with the aria wiring done once. */
export function Field({ label, hideLabel, hint, error, required, children }: WrapProps) {
  const id = useId();
  const msgId = `${id}-msg`;
  const describedBy = error || hint ? msgId : undefined;
  return (
    <div className="flex flex-col gap-1.5">
      <label
        htmlFor={id}
        className={hideLabel
          ? "sr-only"
          : "text-micro uppercase text-content-muted font-medium"}>
        {label}
        {required && <span className="text-danger ml-0.5" aria-hidden="true">*</span>}
      </label>
      {children({ id, describedBy, invalid: !!error })}
      {(error || hint) && (
        <p id={msgId}
          className={`text-xs ${error ? "text-danger" : "text-content-muted"}`}
          // Errors appearing after submit must be announced, not just drawn.
          role={error ? "alert" : undefined}>
          {error || hint}
        </p>
      )}
    </div>
  );
}

interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "size"> {
  size?: "sm" | "md";
  invalid?: boolean;
  /** Numeric fields use tabular figures and the decimal keypad on mobile. */
  numeric?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { size = "md", invalid, numeric, className, ...rest }, ref) {
  return (
    <input
      ref={ref}
      aria-invalid={invalid || undefined}
      inputMode={numeric ? "decimal" : rest.inputMode}
      className={[
        CONTROL, sizeCls(size),
        invalid ? "border-danger" : "border-border focus:border-accent",
        numeric ? "font-mono tabular-nums" : "",
        className ?? "",
      ].filter(Boolean).join(" ")}
      {...rest}
    />
  );
});

interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, "size"> {
  size?: "sm" | "md";
  invalid?: boolean;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { size = "md", invalid, className, children, ...rest }, ref) {
  return (
    <select
      ref={ref}
      aria-invalid={invalid || undefined}
      className={[
        CONTROL, sizeCls(size), "appearance-none pr-7 cursor-pointer",
        invalid ? "border-danger" : "border-border focus:border-accent",
        // Chevron drawn as a background image so the control stays a native
        // <select> — native is keyboard- and screen-reader-correct for free,
        // and on mobile it opens the platform picker.
        "bg-[url('data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22 fill=%22none%22 stroke=%22%2394a3b8%22 stroke-width=%221.5%22><path d=%22m4 6 4 4 4-4%22/></svg>')] bg-no-repeat bg-[right_0.5rem_center] bg-[length:14px_14px]",
        className ?? "",
      ].filter(Boolean).join(" ")}
      {...rest}>
      {children}
    </select>
  );
});
