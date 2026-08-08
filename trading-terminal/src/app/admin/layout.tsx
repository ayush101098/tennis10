import type { Metadata } from "next";

/**
 * Operator console — kept out of the index. robots.txt already disallows it,
 * but a disallowed URL can still be indexed from an external link; the meta
 * directive is what actually prevents it appearing in results.
 */
export const metadata: Metadata = {
  title: "Admin",
  robots: { index: false, follow: false, nocache: true },
};

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return children;
}
