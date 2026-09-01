import type { SVGProps } from "react";

/**
 * The icon set.
 *
 * Replaces emoji used as UI iconography (164 of them in the original code).
 * Emoji render differently on every platform, cannot take a colour, carry no
 * accessible name, and are the clearest single tell of a generated interface.
 *
 * Every icon is 16px on a 16px grid, 1.5px stroke, and inherits `currentColor`
 * so it takes the colour of whatever it sits in.
 *
 * Decorative by default (`aria-hidden`). Pass a `title` when the icon is the
 * only thing conveying meaning, and it becomes a labelled `img` to a screen
 * reader instead.
 */

export type IconName =
  | "trophy" | "gem" | "eye" | "alert" | "target" | "scale" | "lock"
  | "live" | "clock" | "check" | "close" | "plus" | "minus"
  | "chevronDown" | "chevronRight" | "arrowRight" | "external"
  | "trend" | "info" | "search" | "menu";

const PATHS: Record<IconName, React.ReactNode> = {
  trophy: <><path d="M5 3h6v4a3 3 0 0 1-6 0V3Z" /><path d="M5 4H3v1a2 2 0 0 0 2 2M11 4h2v1a2 2 0 0 1-2 2" /><path d="M8 10v2M6 13h4" /></>,
  gem: <><path d="M4 3h8l2 3-6 7-6-7 2-3Z" /><path d="M2 6h12M6 3 4.5 6 8 13M10 3l1.5 3L8 13" /></>,
  eye: <><path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8Z" /><circle cx="8" cy="8" r="2" /></>,
  alert: <><path d="M8 2.5 14.5 13.5h-13L8 2.5Z" /><path d="M8 6.5v3M8 11.5h.01" /></>,
  target: <><circle cx="8" cy="8" r="5.5" /><circle cx="8" cy="8" r="2" /><path d="M8 1v1.5M8 13.5V15M1 8h1.5M13.5 8H15" /></>,
  scale: <><path d="M8 2.5v11M4 4.5h8" /><path d="M3 5 1 9.5h4L3 5ZM13 5l-2 4.5h4L13 5Z" /><path d="M5.5 13.5h5" /></>,
  lock: <><rect x="3" y="7" width="10" height="6.5" rx="1" /><path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" /></>,
  live: <><circle cx="8" cy="8" r="2.5" /><path d="M4.4 4.4a5 5 0 0 0 0 7.2M11.6 4.4a5 5 0 0 1 0 7.2" /></>,
  clock: <><circle cx="8" cy="8" r="5.75" /><path d="M8 4.75V8l2.25 1.5" /></>,
  check: <path d="m3 8.5 3.25 3.25L13 5" />,
  close: <path d="M4 4l8 8M12 4l-8 8" />,
  plus: <path d="M8 3.5v9M3.5 8h9" />,
  minus: <path d="M3.5 8h9" />,
  chevronDown: <path d="m4 6 4 4 4-4" />,
  chevronRight: <path d="m6 4 4 4-4 4" />,
  arrowRight: <><path d="M2.5 8h11" /><path d="m9.5 4 4 4-4 4" /></>,
  external: <><path d="M9 3h4v4" /><path d="M13 3 7.5 8.5" /><path d="M12 9.5V13H3V4h3.5" /></>,
  trend: <><path d="m2 11 3.5-3.5 2.5 2.5L14 4" /><path d="M10 4h4v4" /></>,
  info: <><circle cx="8" cy="8" r="5.75" /><path d="M8 7.25v3.5M8 5.25h.01" /></>,
  search: <><circle cx="7" cy="7" r="4.25" /><path d="m10.25 10.25 3.25 3.25" /></>,
  menu: <path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11" />,
};

interface Props extends Omit<SVGProps<SVGSVGElement>, "name"> {
  name: IconName;
  size?: number;
  /** Give the icon an accessible name. Omit when adjacent text already says it. */
  title?: string;
}

export default function Icon({ name, size = 16, title, className, ...rest }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      // A decorative icon must be invisible to assistive tech; a meaningful
      // one must be announced. Nothing in between.
      role={title ? "img" : undefined}
      aria-hidden={title ? undefined : true}
      aria-label={title}
      focusable="false"
      {...rest}>
      {PATHS[name]}
    </svg>
  );
}

/**
 * The live indicator. A dot, not an emoji, and it stops moving for anyone who
 * has asked for reduced motion (handled globally in globals.css).
 */
export function LiveDot({ className = "" }: { className?: string }) {
  return (
    <span className={`inline-block w-1.5 h-1.5 rounded-full bg-primary live-dot ${className}`}
      aria-hidden="true" />
  );
}
