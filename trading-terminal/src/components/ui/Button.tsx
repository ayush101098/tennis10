"use client";

import { forwardRef } from "react";
import type { ButtonHTMLAttributes, AnchorHTMLAttributes, ReactNode } from "react";
import Icon, { type IconName } from "@/components/ui/Icon";

/**
 * Button — see DESIGN.md §6.
 *
 * Four variants, three sizes, one set of states. Before this existed the app
 * had 92 hand-styled buttons with no shared height, padding, radius or focus
 * treatment; two buttons doing the same job looked different on two screens.
 *
 * The focus ring is deliberately NOT defined here — it comes from the global
 * `:focus-visible` rule so that every control in the app, including the ones
 * not yet migrated, gets the same one.
 */

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-primary text-bg font-semibold hover:opacity-90 active:opacity-80",
  secondary: "bg-surface text-content border border-border hover:border-border-strong active:bg-elevated",
  ghost: "text-content-muted hover:text-content hover:bg-surface active:bg-elevated",
  danger: "bg-danger text-white font-semibold hover:opacity-90 active:opacity-80",
};

// Touch targets reach 44px on coarse pointers regardless of the visual size —
// a 28px control is fine under a mouse and unusable under a thumb.
const SIZES: Record<Size, string> = {
  sm: "h-7 px-2.5 text-xs [@media(pointer:coarse)]:min-h-[44px]",
  md: "h-9 px-3.5 text-sm [@media(pointer:coarse)]:min-h-[44px]",
  lg: "h-11 px-5 text-base",
};

const BASE =
  "inline-flex items-center justify-center gap-1.5 rounded-sm font-medium whitespace-nowrap " +
  "transition-[background-color,border-color,color,opacity] duration-fast ease-standard " +
  "disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none";

interface Common {
  variant?: Variant;
  size?: Size;
  /** Leading icon. Replaced by a spinner while loading. */
  icon?: IconName;
  /** Trailing icon — for directional affordances only. */
  iconAfter?: IconName;
  loading?: boolean;
  fullWidth?: boolean;
  children?: ReactNode;
}

function Spinner() {
  return (
    <svg width={14} height={14} viewBox="0 0 16 16" className="animate-spin" aria-hidden="true">
      <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="2" opacity="0.25" />
      <path d="M14 8a6 6 0 0 0-6-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function content({ icon, iconAfter, loading, children }: Common) {
  return (
    <>
      {loading ? <Spinner /> : icon ? <Icon name={icon} size={14} /> : null}
      {children}
      {iconAfter && !loading ? <Icon name={iconAfter} size={14} /> : null}
    </>
  );
}

const cls = (p: Common, extra?: string) =>
  [BASE, VARIANTS[p.variant ?? "secondary"], SIZES[p.size ?? "md"],
   p.fullWidth ? "w-full" : "", extra ?? ""].filter(Boolean).join(" ");

type ButtonProps = Common & ButtonHTMLAttributes<HTMLButtonElement>;

const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant, size, icon, iconAfter, loading, fullWidth, children, className, disabled, ...rest }, ref) {
  return (
    <button
      ref={ref}
      // A loading button must not fire twice, and must announce that it is busy.
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cls({ variant, size, fullWidth }, className)}
      {...rest}>
      {content({ icon, iconAfter, loading, children })}
    </button>
  );
});

export default Button;

type LinkProps = Common & AnchorHTMLAttributes<HTMLAnchorElement>;

/**
 * A button that navigates. It is an `<a>`, because a thing that goes somewhere
 * must be openable in a new tab and reachable by a screen reader's link list.
 */
export const ButtonLink = forwardRef<HTMLAnchorElement, LinkProps>(function ButtonLink(
  { variant, size, icon, iconAfter, loading, fullWidth, children, className, ...rest }, ref) {
  return (
    <a ref={ref} className={cls({ variant, size, fullWidth }, className)} {...rest}>
      {content({ icon, iconAfter, loading, children })}
    </a>
  );
});

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  name: IconName;
  /** Required: an icon-only control has no other accessible name. */
  label: string;
  variant?: Variant;
  size?: Size;
}

export function IconButton({ name, label, variant = "ghost", size = "md", className, ...rest }: IconButtonProps) {
  const box = size === "sm" ? "h-7 w-7" : size === "lg" ? "h-11 w-11" : "h-9 w-9";
  return (
    <button
      aria-label={label}
      title={label}
      className={[BASE, VARIANTS[variant], box,
        "[@media(pointer:coarse)]:min-h-[44px] [@media(pointer:coarse)]:min-w-[44px]",
        className ?? ""].filter(Boolean).join(" ")}
      {...rest}>
      <Icon name={name} size={size === "sm" ? 14 : 16} />
    </button>
  );
}
