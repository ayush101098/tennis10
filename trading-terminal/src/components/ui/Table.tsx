"use client";

import type { ReactNode, ThHTMLAttributes, TdHTMLAttributes } from "react";

/**
 * Table primitives — DESIGN.md §9.
 *
 * Tables are the primary interface of this product, so the density, alignment
 * and numeric treatment are fixed here rather than re-decided per screen.
 *
 * The rule that matters most: **numbers are right-aligned, monospace and
 * tabular**. A column of probabilities that does not line up is unreadable at
 * a glance, which is the only way anyone reads a board like this.
 */

export function TableScroll({ children, className }: { children: ReactNode; className?: string }) {
  return (
    // Wide content scrolls INSIDE its own container. The page itself must
    // never scroll sideways — that is the classic mobile break.
    <div className={`overflow-x-auto ${className ?? ""}`}>{children}</div>
  );
}

export function Table({ children, className }: { children: ReactNode; className?: string }) {
  return <table className={`w-full border-collapse ${className ?? ""}`}>{children}</table>;
}

export function THead({ children }: { children: ReactNode }) {
  return <thead className="bg-surface sticky top-0 z-10">{children}</thead>;
}

interface ThProps extends ThHTMLAttributes<HTMLTableCellElement> {
  numeric?: boolean;
}

export function Th({ numeric, className, children, ...rest }: ThProps) {
  return (
    <th
      scope="col"
      className={`px-3 py-2 text-micro uppercase font-semibold text-content-muted border-b border-border ${
        numeric ? "text-right" : "text-left"} ${className ?? ""}`}
      {...rest}>
      {children}
    </th>
  );
}

export function Tr({ children, selected, onClick, className }: {
  children: ReactNode; selected?: boolean; onClick?: () => void; className?: string;
}) {
  return (
    <tr
      onClick={onClick}
      // Selection is marked for assistive tech, not just painted.
      aria-selected={selected}
      className={[
        "border-b border-border last:border-b-0",
        onClick ? "cursor-pointer hover:bg-surface/60" : "",
        selected ? "bg-accent/10 border-l-2 border-l-accent" : "",
        className ?? "",
      ].filter(Boolean).join(" ")}>
      {children}
    </tr>
  );
}

interface TdProps extends TdHTMLAttributes<HTMLTableCellElement> {
  numeric?: boolean;
  strong?: boolean;
}

export function Td({ numeric, strong, className, children, ...rest }: TdProps) {
  return (
    <td
      className={[
        "px-3 py-2 text-sm",
        numeric ? "text-right font-mono tabular-nums" : "text-left",
        strong ? "text-content-strong font-semibold" : "text-content",
        className ?? "",
      ].filter(Boolean).join(" ")}
      {...rest}>
      {children}
    </td>
  );
}

/* ── Numeric formatting (DESIGN.md §9) ────────────────────────────────────
   Centralised so the same quantity is never formatted two ways on two
   screens. */

export const pct = (n: number, dp = 1) => `${(n * 100).toFixed(dp)}%`;

/** Signed values carry their sign — "+4.1%" reads differently from "4.1%". */
export const signedPct = (n: number, dp = 1) =>
  `${n >= 0 ? "+" : "−"}${Math.abs(n * 100).toFixed(dp)}%`;

export const odds = (n: number) => n.toFixed(2);

export const money = (n: number) =>
  n >= 1000 ? `$${Math.round(n).toLocaleString("en-US")}` : `$${Math.round(n)}`;

/** Colour follows the sign — and is never the only signal, since the sign is there too. */
export const signTone = (n: number, floor = 0) =>
  n > floor ? "text-primary" : n > 0 ? "text-accent" : "text-danger";
