"use client";

import type { ReactNode } from "react";
import Icon, { type IconName, LiveDot } from "@/components/ui/Icon";

/**
 * Containers, states and small display primitives — DESIGN.md §8, §9, §11.
 *
 * `Panel` is the app's one container. It replaces the pattern of every screen
 * inventing its own bordered box, and it is deliberately the ONLY thing with a
 * border and radius in the page flow: no nested panels, no shadow. When
 * everything is boxed, nothing is grouped — which is what made the original
 * screens read as generated.
 */

export type Tone = "neutral" | "primary" | "accent" | "warning" | "danger";

const HEADER_TONE: Record<Tone, string> = {
  neutral: "text-content-strong",
  primary: "text-primary",
  accent: "text-accent",
  warning: "text-warning",
  danger: "text-danger",
};

const EDGE_TONE: Record<Tone, string> = {
  neutral: "border-border",
  primary: "border-primary/40",
  accent: "border-accent/40",
  warning: "border-warning/40",
  danger: "border-danger/40",
};

interface PanelProps {
  title?: ReactNode;
  icon?: IconName;
  tone?: Tone;
  /** Right-aligned metadata in the header — counts, timestamps, controls. */
  meta?: ReactNode;
  live?: boolean;
  children: ReactNode;
  className?: string;
  id?: string;
}

export function Panel({ title, icon, tone = "neutral", meta, live, children, className, id }: PanelProps) {
  return (
    <section
      id={id}
      className={`border ${EDGE_TONE[tone]} rounded-md bg-surface/30 overflow-hidden ${className ?? ""}`}>
      {title && (
        <header className={`flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-4 py-2.5 border-b ${EDGE_TONE[tone]} ${tone === "neutral" ? "bg-surface/60" : ""}`}>
          <h2 className={`flex items-center gap-2 text-sm font-semibold tracking-wide ${HEADER_TONE[tone]}`}>
            {icon && <Icon name={icon} size={15} />}
            {title}
            {live && <LiveDot />}
          </h2>
          {meta && <div className="text-xs text-content-muted">{meta}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

/** A labelled figure. The label is small and quiet; the number is the point. */
export function Stat({ label, value, tone = "neutral", mono = true, className }: {
  label: string; value: ReactNode; tone?: Tone; mono?: boolean; className?: string;
}) {
  const toneCls = tone === "neutral" ? "text-content-strong" : HEADER_TONE[tone];
  return (
    <div className={className}>
      <div className="text-micro uppercase text-content-muted">{label}</div>
      <div className={`${mono ? "font-mono tabular-nums" : ""} text-lg font-semibold ${toneCls}`}>
        {value}
      </div>
    </div>
  );
}

const BADGE_TONE: Record<Tone, string> = {
  neutral: "bg-surface text-content-muted border-border",
  primary: "bg-primary/10 text-primary border-primary/30",
  accent: "bg-accent/10 text-accent border-accent/30",
  warning: "bg-warning/10 text-warning border-warning/30",
  danger: "bg-danger/10 text-danger border-danger/30",
};

export function Badge({ children, tone = "neutral", icon }: {
  children: ReactNode; tone?: Tone; icon?: IconName;
}) {
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-sm border text-micro uppercase font-semibold ${BADGE_TONE[tone]}`}>
      {icon && <Icon name={icon} size={11} />}
      {children}
    </span>
  );
}

/* ── The three states every data surface must define (DESIGN.md §11) ── */

export function LoadingState({ label = "Loading…", rows = 3 }: { label?: string; rows?: number }) {
  return (
    <div className="p-4" role="status" aria-live="polite" aria-busy="true">
      <span className="sr-only">{label}</span>
      <div className="flex flex-col gap-2">
        {Array.from({ length: rows }).map((_, i) => (
          // Skeletons match the shape of what is coming, so nothing jumps when
          // it arrives. A bare spinner in an empty panel tells the user nothing
          // about what to expect.
          <div key={i} className="flex items-center gap-3">
            <div className="h-3 w-12 rounded-sm bg-border animate-pulse" />
            <div className="h-3 flex-1 rounded-sm bg-border animate-pulse" />
            <div className="h-3 w-10 rounded-sm bg-border animate-pulse" />
          </div>
        ))}
      </div>
    </div>
  );
}

export function EmptyState({ title, body, action }: {
  title: string; body?: string; action?: ReactNode;
}) {
  return (
    <div className="px-6 py-10 text-center">
      <p className="text-base font-semibold text-content-strong">{title}</p>
      {body && <p className="mt-1 text-sm text-content-muted max-w-[46ch] mx-auto">{body}</p>}
      {action && <div className="mt-4 flex justify-center">{action}</div>}
    </div>
  );
}

export function ErrorState({ title, body, action }: {
  title: string; body?: string; action?: ReactNode;
}) {
  return (
    <div className="px-6 py-10 text-center" role="alert">
      <p className="flex items-center justify-center gap-1.5 text-base font-semibold text-warning">
        <Icon name="alert" size={15} />
        {title}
      </p>
      {body && <p className="mt-1 text-sm text-content-muted max-w-[52ch] mx-auto">{body}</p>}
      {action && <div className="mt-4 flex justify-center">{action}</div>}
    </div>
  );
}

/**
 * Missing data. An em-dash, never `0`, `N/A` or an empty cell — a zero is a
 * measurement and "we don't know" is not.
 */
export const NoValue = () => <span className="text-content-muted" aria-label="no value">—</span>;
