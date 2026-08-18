"""Layer 1 — perception: ball tracking, player tracking, court calibration.

Runs on any frame sequence (synthetic or a real video decoded by pipeline.py).
Everything emits a per-detection confidence, because the PRD makes confidence a
P0 requirement and downstream layers must be able to exclude low-confidence
frames rather than silently trust them.

Design notes / honest limits:
  • Ball detection is colour-blob based (bright yellow-green). This is robust on
    controlled/synthetic input and decent on clean broadcast, but a real
    broadcast ball needs a trained small-object detector (e.g. TrackNet). The
    blob path is deliberately swappable — see detect_ball().
  • Players are found by background subtraction (median background), which suits
    a fixed camera. Broadcast cuts need a learned detector (YOLO) — same swap
    point in detect_players().
  • Court calibration fits a homography from the court quad to a canonical
    top-down court, giving metric (x,y) so speeds/coverage are in real units.
"""

from __future__ import annotations

import cv2
import numpy as np

# canonical top-down singles court in millimetres (ITF): 8230 wide x 23770 long
COURT_MM = np.float32([[0, 0], [8230, 0], [8230, 23770], [0, 23770]])


# ── court calibration ────────────────────────────────────────────────────────

def calibrate_court(frames: list[np.ndarray]) -> dict:
    """Fit image->court homography from the court surface quad. Returns
    {H, corners_img, conf}. conf drops if we can't find a clean 4-gon."""
    bg = np.median(np.stack(frames[:: max(1, len(frames) // 15)]), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(bg, cv2.COLOR_RGB2HSV)
    r, g, b = bg[..., 0].astype(int), bg[..., 1].astype(int), bg[..., 2].astype(int)
    # court surface: warm/mid brightness, r>g>b (clay/hard), not the green surround
    mask = ((r > 110) & (r < 210) & (r > g) & (g > b) & (b < 140)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"H": None, "corners_img": None, "conf": 0.0}
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) != 4:
        hull = cv2.convexHull(c)
        approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
    conf = 0.9 if len(approx) == 4 else 0.4
    if len(approx) != 4:
        # fall back to bounding quad of the contour
        x, y, w, h = cv2.boundingRect(c)
        approx = np.array([[[x, y + h]], [[x + w, y + h]], [[x + w, y]], [[x, y]]])
    pts = approx.reshape(-1, 2).astype(np.float32)
    pts = _order_quad(pts)                      # near-left, near-right, far-right, far-left
    Hmat, _ = cv2.findHomography(pts, COURT_MM)
    return {"H": Hmat, "corners_img": pts.tolist(), "conf": float(conf)}


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 pts as near-left, near-right, far-right, far-left (image y down)."""
    ys = pts[:, 1]
    near = pts[np.argsort(ys)[2:]]              # two largest y = nearer baseline
    far = pts[np.argsort(ys)[:2]]
    near = near[np.argsort(near[:, 0])]         # left, right
    far = far[np.argsort(far[:, 0])]            # left, right
    return np.float32([near[0], near[1], far[1], far[0]])


def to_court_mm(H: np.ndarray | None, x: float, y: float):
    """Project an image point to court millimetres, or None if uncalibrated."""
    if H is None:
        return None
    p = H @ np.array([x, y, 1.0])
    if abs(p[2]) < 1e-9:
        return None
    return (float(p[0] / p[2]), float(p[1] / p[2]))


# ── ball detection ───────────────────────────────────────────────────────────

def detect_ball(frame: np.ndarray) -> dict:
    """Bright yellow-green blob -> (x, y, conf). conf reflects blob quality."""
    r, g, b = frame[..., 0].astype(int), frame[..., 1].astype(int), frame[..., 2].astype(int)
    mask = ((g > 160) & (b > 160) & (r < 170) & (np.abs(g - b) < 90)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if 2 <= cv2.contourArea(c) <= 400]
    if not cnts:
        return {"xy": None, "conf": 0.0}
    c = max(cnts, key=cv2.contourArea)
    (x, y), rad = cv2.minEnclosingCircle(c)
    area = cv2.contourArea(c)
    circ = area / (np.pi * rad * rad + 1e-6)          # 1.0 = perfect disc
    conf = float(max(0.0, min(1.0, circ)) * (1.0 if area <= 200 else 0.6))
    return {"xy": (float(x), float(y)), "conf": conf}


def smooth_track(track: list[dict], max_gap: int = 5) -> list[dict]:
    """Fill short gaps in a per-frame xy track by linear interpolation, tagging
    interpolated frames with reduced confidence. Long gaps stay None."""
    out = [dict(t) for t in track]
    n = len(out)
    i = 0
    while i < n:
        if out[i]["xy"] is None:
            j = i
            while j < n and out[j]["xy"] is None:
                j += 1
            a = i - 1
            if a >= 0 and j < n and (j - a) <= max_gap + 1:
                pa, pb = np.array(out[a]["xy"]), np.array(out[j]["xy"])
                for k in range(i, j):
                    t = (k - a) / (j - a)
                    out[k]["xy"] = tuple((pa + t * (pb - pa)).tolist())
                    out[k]["conf"] = 0.35
                    out[k]["interp"] = True
            i = j
        else:
            i += 1
    return out


# ── player detection & tracking ──────────────────────────────────────────────

def _median_bg(frames: list[np.ndarray]) -> np.ndarray:
    idx = np.linspace(0, len(frames) - 1, min(len(frames), 25)).astype(int)
    return np.median(np.stack([frames[i] for i in idx]), axis=0).astype(np.uint8)


def detect_players(frames: list[np.ndarray]) -> list[list[dict]]:
    """Per-frame list of up to 2 players: {id, bbox, foot(px), centroid, conf}.
    Fixed-camera background subtraction + nearest-centroid ID assignment."""
    bg = _median_bg(frames)
    prev = {}                       # id -> centroid
    next_id = 0
    out = []
    for fr in frames:
        diff = cv2.absdiff(fr, bg).sum(axis=2).astype(np.uint8)
        fgmask = (diff > 60).astype(np.uint8) * 255
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 120:                      # ignore ball / noise
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h < 1.1 * w:                  # players are taller than wide-ish
                pass
            cx, cy = x + w / 2, y + h / 2
            blobs.append({"bbox": [x, y, w, h], "centroid": (cx, cy),
                          "foot": (cx, y + h), "area": a})
        blobs.sort(key=lambda d: -d["area"])
        blobs = blobs[:2]
        # assign IDs by nearest previous centroid
        assigned = {}
        used = set()
        for pid, pc in prev.items():
            best, bd = None, 1e9
            for k, bl in enumerate(blobs):
                if k in used:
                    continue
                d = np.hypot(bl["centroid"][0] - pc[0], bl["centroid"][1] - pc[1])
                if d < bd:
                    bd, best = d, k
            if best is not None and bd < 160:
                assigned[best] = pid
                used.add(best)
        frame_players = []
        for k, bl in enumerate(blobs):
            if k in assigned:
                pid = assigned[k]
            else:
                pid = next_id; next_id += 1
            prev[pid] = bl["centroid"]
            frame_players.append({"id": int(pid), "bbox": bl["bbox"],
                                  "foot": bl["foot"], "centroid": bl["centroid"],
                                  "conf": float(min(1.0, bl["area"] / 800.0))})
        out.append(frame_players)
    # keep only the two most-persistent ids as the two players
    from collections import Counter
    freq = Counter(p["id"] for fp in out for p in fp)
    keep = {pid for pid, _ in freq.most_common(2)}
    remap = {pid: i for i, pid in enumerate(sorted(keep))}
    for fp in out:
        fp[:] = [{**p, "id": remap[p["id"]]} for p in fp if p["id"] in keep]
    return out


def run_perception(frames: list[np.ndarray]) -> dict:
    """Full Layer-1 pass. Returns court + per-frame ball track + players."""
    court = calibrate_court(frames)
    ball_raw = [detect_ball(f) for f in frames]
    ball = smooth_track(ball_raw)
    players = detect_players(frames)
    return {"court": court, "ball": ball, "players": players,
            "n_frames": len(frames)}
