"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { IconButton } from "@/components/ui/Button";

/**
 * Modal — the app's one dialog shell.
 *
 * A dialog is the easiest thing in a UI to get wrong for keyboard and screen
 * reader users, so the behaviour lives here once: focus moves in on open,
 * is trapped while open, Escape closes, the page behind cannot scroll, and
 * focus returns to whatever opened it. Screens that hand-rolled this got some
 * of it and never all of it.
 */

interface Props {
  open: boolean;
  onClose: () => void;
  title: string;
  /** Hide the visible title but keep the accessible name. */
  hideTitle?: boolean;
  children: ReactNode;
  footer?: ReactNode;
  size?: "sm" | "md" | "lg";
}

const WIDTH = { sm: "max-w-[420px]", md: "max-w-[640px]", lg: "max-w-[880px]" };

export default function Modal({ open, onClose, title, hideTitle, children, footer, size = "md" }: Props) {
  const panelRef = useRef<HTMLDivElement>(null);
  const restoreTo = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;

    restoreTo.current = document.activeElement as HTMLElement | null;

    // The page behind must not scroll under the dialog.
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const focusables = () =>
      Array.from(panelRef.current?.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ) ?? []).filter(el => el.offsetParent !== null);

    // Move focus into the dialog, not just visually onto it.
    (focusables()[0] ?? panelRef.current)?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.stopPropagation(); onClose(); return; }
      if (e.key !== "Tab") return;
      const list = focusables();
      if (!list.length) return;
      const first = list[0], last = list[list.length - 1];
      // Wrap, so Tab can never land behind the dialog.
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };

    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      document.body.style.overflow = prevOverflow;
      restoreTo.current?.focus?.();
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-bg/80"
      // A click on the backdrop is a dismissal; a click inside is not.
      onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        className={`w-full ${WIDTH[size]} max-h-[calc(100dvh-2rem)] flex flex-col bg-elevated border border-border rounded-md shadow-overlay`}>
        <header className="flex items-center justify-between gap-3 px-4 py-3 border-b border-border shrink-0">
          <h2 className={hideTitle ? "sr-only" : "text-base font-semibold text-content-strong"}>{title}</h2>
          <IconButton name="close" label="Close dialog" size="sm" onClick={onClose} className="-mr-1" />
        </header>
        <div className="overflow-y-auto p-4">{children}</div>
        {footer && <footer className="px-4 py-3 border-t border-border shrink-0">{footer}</footer>}
      </div>
    </div>
  );
}

/**
 * Toast — transient confirmation.
 *
 * `role="status"` rather than `alert`: a confirmation should be announced when
 * the screen reader next pauses, not interrupt what it is mid-sentence on.
 */
export function Toast({ message, tone = "neutral", onDismiss }: {
  message: string; tone?: "neutral" | "primary" | "danger"; onDismiss?: () => void;
}) {
  const toneCls = tone === "primary" ? "border-primary/40 text-primary"
    : tone === "danger" ? "border-danger/40 text-danger"
    : "border-border text-content";
  return (
    <div role="status" aria-live="polite"
      className={`fixed bottom-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-4 py-2.5 bg-elevated border rounded-md shadow-overlay text-sm ${toneCls}`}>
      {message}
      {onDismiss && <IconButton name="close" label="Dismiss" size="sm" onClick={onDismiss} />}
    </div>
  );
}
