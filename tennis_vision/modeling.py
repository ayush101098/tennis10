"""Layer 2 — modeling: shots, strokes, rally stats, match-state.

Consumes Layer-1 perception (ball track + players + court homography) and turns
pixels into tennis. Two principles carried straight from the PRD:

  • Every derived stat carries a confidence, inherited from the detections it was
    built on, so the application layer can grey-out or exclude weak numbers.
  • Outputs are framed honestly. The "match-state / momentum" number is a toy
    indicator from shot cadence, NOT a validated win model — it is labelled as
    such, because over-claiming predictive certainty is called out as a risk.

Stroke labels are heuristic (serve = first shot; forehand/backhand from hitter
side + ball lateral direction). Real technique/stroke classification needs a
pose model trained on labelled strokes — this is the swap point, and the
function signatures already take player keypoints when available.
"""

from __future__ import annotations

import numpy as np

from .perception import to_court_mm


def _ball_xy(ball):
    return np.array([[b["xy"][0], b["xy"][1]] if b["xy"] else [np.nan, np.nan]
                     for b in ball], float)


def detect_hits(ball: list[dict], players: list[list[dict]], fps: int) -> list[dict]:
    """A hit = a significant reversal of the ball's along-court (vertical-image)
    velocity. Returns [{frame, hitter_id, conf}]. Robust to short gaps."""
    xy = _ball_xy(ball)
    vy = np.gradient(xy[:, 1])
    hits = []
    last = -10
    for i in range(2, len(vy) - 2):
        if np.isnan(vy[i - 1]) or np.isnan(vy[i + 1]):
            continue
        # sign flip in vertical velocity with enough magnitude on both sides
        if vy[i - 1] * vy[i + 1] < 0 and (abs(vy[i - 1]) + abs(vy[i + 1])) > 2.5:
            if i - last < max(4, fps // 6):
                continue
            last = i
            hitter, conf = _nearest_player(xy[i], players[i] if i < len(players) else [])
            hits.append({"frame": int(i), "hitter_id": hitter,
                         "conf": float(min(ball[i]["conf"] + 0.2, 1.0) * conf)})
    return hits


def _nearest_player(ball_xy, frame_players):
    if ball_xy is None or np.isnan(ball_xy).any() or not frame_players:
        return None, 0.3
    d = [(np.hypot(p["foot"][0] - ball_xy[0], p["foot"][1] - ball_xy[1]), p)
         for p in frame_players]
    d.sort(key=lambda t: t[0])
    return int(d[0][1]["id"]), 0.9 if d[0][0] < 220 else 0.5


def classify_strokes(hits: list[dict], ball: list[dict]) -> list[dict]:
    """Label each hit. First = serve; then forehand/backhand from ball lateral
    direction relative to the hitter (heuristic, right-handed assumption)."""
    xy = _ball_xy(ball)
    out = []
    for k, h in enumerate(hits):
        f = h["frame"]
        if k == 0:
            label = "serve"
        else:
            dx = 0.0
            if f + 3 < len(xy) and not np.isnan(xy[f + 3, 0]) and not np.isnan(xy[f, 0]):
                dx = xy[f + 3, 0] - xy[f, 0]
            # ball moving to hitter's left vs right -> BH/FH (rough, side-dependent)
            side = h["hitter_id"] or 0
            fh = (dx > 0) if side == 0 else (dx < 0)
            label = "forehand" if fh else "backhand"
        out.append({**h, "stroke": label})
    return out


def rally_stats(perc: dict, hits: list[dict], fps: int) -> dict:
    """Aggregate rally-level numbers in real units where the court is calibrated."""
    H = perc["court"]["H"]
    ball = perc["ball"]
    xy = _ball_xy(ball)

    # ball speed between consecutive good detections, in km/h via homography
    speeds = []
    for i in range(1, len(xy)):
        if np.isnan(xy[i]).any() or np.isnan(xy[i - 1]).any():
            continue
        a = to_court_mm(H, *xy[i - 1]); b = to_court_mm(H, *xy[i])
        if not a or not b:
            continue
        dmm = np.hypot(b[0] - a[0], b[1] - a[1])
        speeds.append((dmm / 1000.0) * fps * 3.6)      # mm/frame -> m/s -> km/h
    speeds = [s for s in speeds if s < 300]            # drop homography blowups

    # per-player distance covered (metres) + coverage heatmap points (court mm)
    coverage = {0: 0.0, 1: 0.0}
    heat = {0: [], 1: []}
    last_pos = {}
    for fp in perc["players"]:
        for p in fp:
            cm = to_court_mm(H, *p["foot"])
            if not cm:
                continue
            heat[p["id"]].append(cm)
            if p["id"] in last_pos:
                d = np.hypot(cm[0] - last_pos[p["id"]][0], cm[1] - last_pos[p["id"]][1])
                if d < 2000:                            # ignore ID-swap jumps
                    coverage[p["id"]] += d / 1000.0
            last_pos[p["id"]] = cm

    dur = perc["n_frames"] / fps
    ball_conf = float(np.mean([b["conf"] for b in ball]))
    return {
        "n_shots": len(hits),
        "rally_seconds": round(dur, 2),
        "shots_per_second": round(len(hits) / dur, 2) if dur else 0.0,
        "ball_speed_kmh": {
            "max": round(max(speeds), 1) if speeds else None,
            "mean": round(float(np.mean(speeds)), 1) if speeds else None,
        },
        "player_distance_m": {str(k): round(v, 1) for k, v in coverage.items()},
        "heatmap": {str(k): v for k, v in heat.items()},
        "confidence": {
            "ball_track": round(ball_conf, 2),
            "court_calibration": round(perc["court"]["conf"], 2),
            "overall": round(ball_conf * perc["court"]["conf"], 2),
        },
    }


def match_state(strokes: list[dict], stats: dict) -> dict:
    """TOY momentum indicator (NOT a validated win model): who is dictating,
    from who hits more + rally tempo. Explicitly low-authority per the PRD."""
    from collections import Counter
    by = Counter(s["hitter_id"] for s in strokes if s["hitter_id"] is not None)
    total = sum(by.values()) or 1
    lead = max(by, key=by.get) if by else None
    share = by[lead] / total if lead is not None else 0.5
    return {
        "dictating_player": (None if lead is None else int(lead)),
        "shot_share": {str(k): round(v / total, 2) for k, v in by.items()},
        "tempo_shots_per_s": stats["shots_per_second"],
        "note": "toy momentum from shot cadence — not a validated win-probability model",
        "confidence": 0.35,
    }


def run_modeling(perc: dict, fps: int) -> dict:
    hits = detect_hits(perc["ball"], perc["players"], fps)
    strokes = classify_strokes(hits, perc["ball"])
    stats = rally_stats(perc, hits, fps)
    state = match_state(strokes, stats)
    return {"hits": hits, "strokes": strokes, "stats": stats, "match_state": state}
