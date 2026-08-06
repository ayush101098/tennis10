"use client";

import { useEffect, useState } from "react";
import QRCode from "qrcode";

/**
 * QR for a payment target — a PayPal.me link or a wallet address.
 *
 * Encoded in the browser to a data: URI rather than pointing an <img> at a
 * QR-as-a-service host: a payment address is the one thing on this page that
 * must not be fetched from a third party who could serve a different image,
 * and it keeps the payee out of someone else's request logs.
 *
 * Rendered light-on-white deliberately — phone scanners are far more reliable
 * against a white quiet zone than against the terminal's dark background.
 */
export default function QrCode({ value, size = 132, label }: {
  value: string;
  size?: number;
  label?: string;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    QRCode.toDataURL(value, {
      width: size * 2,          // 2x so it stays sharp on retina phones
      margin: 2,                // the quiet zone scanners need
      errorCorrectionLevel: "M",
      color: { dark: "#0a0e17", light: "#ffffff" },
    })
      .then(url => { if (alive) { setSrc(url); setFailed(false); } })
      .catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, [value, size]);

  // Never leave a blank square where a payment target should be — the text
  // form below it is always the fallback.
  if (failed) return null;

  return (
    <figure className="flex flex-col items-center gap-1.5">
      <div className="rounded bg-white p-1.5" style={{ width: size + 12, height: size + 12 }}>
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element -- data: URI, nothing to optimise
          <img src={src} alt={label ? `QR code — ${label}` : "Payment QR code"}
            width={size} height={size} style={{ width: size, height: size, display: "block" }} />
        ) : (
          <div style={{ width: size, height: size }} className="animate-pulse bg-slate-200 rounded" />
        )}
      </div>
      {label && <figcaption className="text-[9px] text-terminal-muted">{label}</figcaption>}
    </figure>
  );
}
