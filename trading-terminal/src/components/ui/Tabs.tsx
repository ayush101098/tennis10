"use client";

import { useRef, type ReactNode } from "react";
import Icon, { type IconName } from "@/components/ui/Icon";

/**
 * Tabs — DESIGN.md §10, and the keyboard contract from the WAI-ARIA tabs
 * pattern: arrow keys move between tabs, Home/End jump to the ends, and only
 * the active tab is in the tab order. Without that, a tab strip is just a row
 * of buttons that looks like tabs.
 */

export interface TabItem {
  id: string;
  label: string;
  icon?: IconName;
  /** Optional count shown after the label — e.g. how many rows are behind it. */
  count?: number;
}

export function Tabs({ items, active, onChange, className }: {
  items: TabItem[]; active: string; onChange: (id: string) => void; className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  const onKeyDown = (e: React.KeyboardEvent) => {
    const i = items.findIndex(t => t.id === active);
    let next = -1;
    if (e.key === "ArrowRight") next = (i + 1) % items.length;
    else if (e.key === "ArrowLeft") next = (i - 1 + items.length) % items.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = items.length - 1;
    if (next < 0) return;
    e.preventDefault();
    onChange(items[next].id);
    ref.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus();
  };

  return (
    <div
      ref={ref}
      role="tablist"
      onKeyDown={onKeyDown}
      className={`flex items-center gap-1 border-b border-border ${className ?? ""}`}>
      {items.map(t => {
        const on = t.id === active;
        return (
          <button
            key={t.id}
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={on}
            aria-controls={`panel-${t.id}`}
            // Only the selected tab is tabbable; arrows move within the strip.
            tabIndex={on ? 0 : -1}
            onClick={() => onChange(t.id)}
            className={[
              "inline-flex items-center gap-1.5 px-3 h-9 text-sm font-medium rounded-t-sm -mb-px border-b-2",
              "transition-colors duration-fast ease-standard",
              "[@media(pointer:coarse)]:min-h-[44px]",
              on ? "border-primary text-content-strong"
                 : "border-transparent text-content-muted hover:text-content",
            ].join(" ")}>
            {t.icon && <Icon name={t.icon} size={14} />}
            {t.label}
            {t.count !== undefined && (
              <span className="font-mono tabular-nums text-xs text-content-muted">{t.count}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

export function TabPanel({ id, active, children }: { id: string; active: string; children: ReactNode }) {
  if (id !== active) return null;
  return (
    <div role="tabpanel" id={`panel-${id}`} aria-labelledby={`tab-${id}`} tabIndex={0}>
      {children}
    </div>
  );
}

/**
 * Tooltip.
 *
 * CSS-only on hover AND focus, so it is reachable by keyboard. The text is
 * always in the DOM and linked by `aria-describedby` rather than being
 * invented on hover — a tooltip a screen reader cannot read is decoration.
 *
 * Never put essential information here and nowhere else: touch devices have no
 * hover, so anything only in a tooltip does not exist on a phone.
 */
export function Tooltip({ label, children, id }: { label: string; children: ReactNode; id: string }) {
  return (
    <span className="relative inline-flex group">
      <span aria-describedby={id} className="inline-flex">{children}</span>
      <span
        id={id}
        role="tooltip"
        className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 z-30
                   whitespace-nowrap px-2 py-1 rounded-sm bg-elevated border border-border
                   text-xs text-content shadow-overlay
                   opacity-0 group-hover:opacity-100 group-focus-within:opacity-100
                   transition-opacity duration-fast ease-standard">
        {label}
      </span>
    </span>
  );
}
