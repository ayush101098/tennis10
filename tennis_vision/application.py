"""Layer 3 — application: annotated video + self-contained HTML dashboard.

Turns the modeling output into the two artefacts a coach/analyst actually looks
at: an overlay video (court, ball trail, player IDs, live stroke label) and a
one-file HTML report (charts embedded as base64, so it opens anywhere with no
server). Confidence numbers are shown, not hidden, per the PRD.
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np


def annotate(frames, perc, model):
    """Return a new list of RGB frames with perception/modeling overlays."""
    ball, players = perc["ball"], perc["players"]
    corners = perc["court"]["corners_img"]
    stroke_at = {s["frame"]: s for s in model["strokes"]}
    out = []
    trail = []
    last_stroke = None
    for i, fr in enumerate(frames):
        img = fr.copy()
        if corners:
            pts = np.array(corners, np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 120), 1)
        # players
        for p in (players[i] if i < len(players) else []):
            x, y, w, h = p["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 80, 80), 1)
            cv2.putText(img, f"P{p['id']}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 80, 80), 1, cv2.LINE_AA)
        # ball + fading trail
        if ball[i]["xy"]:
            trail.append(ball[i]["xy"])
            trail = trail[-14:]
            for j, (tx, ty) in enumerate(trail):
                a = (j + 1) / len(trail)
                cv2.circle(img, (int(tx), int(ty)), 1, (int(80 * a), int(240 * a), int(240 * a)), -1)
            bx, by = ball[i]["xy"]
            col = (60, 240, 240) if not ball[i].get("interp") else (120, 120, 200)
            cv2.circle(img, (int(bx), int(by)), 5, col, 1)
        # stroke label (hold for ~0.4s)
        if i in stroke_at:
            last_stroke = (i, stroke_at[i])
        if last_stroke and i - last_stroke[0] < 12:
            s = last_stroke[1]
            cv2.putText(img, f"{s['stroke'].upper()} (P{s['hitter_id']})", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 240, 240), 2, cv2.LINE_AA)
        # HUD
        st = model["stats"]
        hud = f"shots {st['n_shots']}  max {st['ball_speed_kmh']['max']} km/h  conf {st['confidence']['overall']}"
        cv2.putText(img, hud, (12, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (235, 235, 235), 1, cv2.LINE_AA)
        out.append(img)
    return out


def write_video(frames, path, fps):
    import imageio
    imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
    return path


# ── charts (matplotlib -> base64 png) ────────────────────────────────────────

def _png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _charts(perc, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .perception import to_court_mm, COURT_MM
    imgs = {}

    # 1. ball trajectory in court coords
    H = perc["court"]["H"]
    pts = [to_court_mm(H, b["xy"][0], b["xy"][1]) for b in perc["ball"] if b["xy"]]
    pts = [p for p in pts if p]
    fig, ax = plt.subplots(figsize=(3.2, 5))
    cx = [0, 8230, 8230, 0, 0]; cy = [0, 0, 23770, 23770, 0]
    ax.plot(cx, cy, "-", color="#888"); ax.plot([0, 8230], [11885, 11885], "--", color="#bbb")
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="#f0c", lw=1)
    ax.set_title("Ball path (court mm)"); ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    imgs["trajectory"] = _png_b64(fig)

    # 2. speed over time (per-frame ball speed already summarised; recompute quick)
    fig, ax = plt.subplots(figsize=(5, 2.4))
    xy = np.array([[b["xy"][0], b["xy"][1]] if b["xy"] else [np.nan, np.nan]
                   for b in perc["ball"]], float)
    v = np.hypot(np.gradient(xy[:, 0]), np.gradient(xy[:, 1]))
    ax.plot(v, color="#2a8");
    for h in model["hits"]:
        ax.axvline(h["frame"], color="#f55", alpha=0.4, lw=1)
    ax.set_title("Ball pixel-speed (red = detected hits)"); ax.set_xlabel("frame")
    imgs["speed"] = _png_b64(fig)

    # 3. coverage heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(5, 4))
    for pid, ax in zip(["0", "1"], axes):
        h = model["stats"]["heatmap"].get(pid, [])
        ax.plot(cx, cy, "-", color="#aaa")
        if h:
            ax.scatter([p[0] for p in h], [p[1] for p in h], s=6, alpha=0.3,
                       color="#e33" if pid == "0" else "#dd0")
        ax.set_title(f"P{pid} coverage"); ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
    imgs["coverage"] = _png_b64(fig)
    return imgs


def build_dashboard(perc, model, validation, path):
    imgs = _charts(perc, model)
    st = model["stats"]; ms = model["match_state"]
    rows = "".join(
        f"<tr><td>{i+1}</td><td>{s['stroke']}</td><td>P{s['hitter_id']}</td>"
        f"<td>frame {s['frame']}</td><td>{s['conf']:.2f}</td></tr>"
        for i, s in enumerate(model["strokes"]))
    val_html = ""
    if validation:
        val_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in validation.items())
        val_html = f"<h2>Validation vs ground truth</h2><table>{val_html}</table>"
    html = f"""<!doctype html><meta charset=utf8><title>Tennis Vision — rally report</title>
<style>
body{{font:14px/1.5 system-ui;margin:0;background:#0f1216;color:#e6e6e6}}
.wrap{{max-width:900px;margin:0 auto;padding:24px}}
h1{{font-size:20px}} h2{{font-size:15px;color:#8fd;margin-top:28px}}
.kpis{{display:flex;gap:12px;flex-wrap:wrap}}
.kpi{{background:#1a1f26;border:1px solid #2a2f36;border-radius:8px;padding:12px 16px;min-width:120px}}
.kpi b{{display:block;font-size:22px;color:#6ee}} .kpi span{{color:#9aa}}
table{{border-collapse:collapse;width:100%;margin-top:8px}}
td,th{{border:1px solid #2a2f36;padding:5px 8px;text-align:left}} th{{color:#9aa}}
img{{max-width:100%;background:#fff;border-radius:6px;margin-top:8px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.note{{color:#c96;font-size:12px}}
</style>
<div class=wrap>
<h1>🎾 Tennis Vision — rally report</h1>
<div class=kpis>
 <div class=kpi><b>{st['n_shots']}</b><span>shots</span></div>
 <div class=kpi><b>{st['rally_seconds']}s</b><span>rally length</span></div>
 <div class=kpi><b>{st['ball_speed_kmh']['max']}</b><span>max km/h</span></div>
 <div class=kpi><b>{st['player_distance_m'].get('0','–')}</b><span>P0 run (m)</span></div>
 <div class=kpi><b>{st['player_distance_m'].get('1','–')}</b><span>P1 run (m)</span></div>
 <div class=kpi><b>{st['confidence']['overall']}</b><span>overall conf</span></div>
</div>
<h2>Match-state (toy)</h2>
<p>Dictating: <b>P{ms['dictating_player']}</b> · tempo {ms['tempo_shots_per_s']} shots/s
 · <span class=note>{ms['note']}</span></p>
<div class=grid>
 <div><h2>Ball path</h2><img src="data:image/png;base64,{imgs['trajectory']}"></div>
 <div><h2>Speed &amp; hits</h2><img src="data:image/png;base64,{imgs['speed']}"></div>
</div>
<h2>Court coverage</h2><img src="data:image/png;base64,{imgs['coverage']}">
<h2>Shots</h2><table><tr><th>#</th><th>stroke</th><th>player</th><th>when</th><th>conf</th></tr>{rows}</table>
{val_html}
<h2>Confidence</h2>
<p class=note>Ball track {st['confidence']['ball_track']} · court calibration
 {st['confidence']['court_calibration']} · overall {st['confidence']['overall']}.
 Accuracy depends on input: this run is {'synthetic (ground-truth validated)' if validation else 'real video (blob detectors — see limits in perception.py)'}.</p>
</div>"""
    with open(path, "w") as f:
        f.write(html)
    return path
