"""Layer 0 — synthetic tennis rally generator (ground truth).

The PRD insists accuracy be *measured against labeled data*, not asserted. Real
broadcast video has no per-frame ground truth lying around, so this module
renders a physically-plausible rally in a broadcast-style perspective and emits
the exact ball / player / court / hit labels alongside the pixels. The
perception layer is then scored against these labels (see pipeline.validate),
which is how we honour "report accuracy per configuration, don't gloss over it".

Everything is deterministic given a seed, so runs are reproducible.

    frames, truth = render_rally(seed=7)
    #  frames: list[np.uint8 HxWx3]   truth: dict with per-frame labels

Real video? `pipeline.py --video path.mp4` skips this and runs perception
straight on the file — the synthetic path is the self-contained default.
"""

from __future__ import annotations

import numpy as np

W, H = 960, 540
FPS = 30

# Broadcast-style court trapezoid in image space (far baseline narrower than
# near baseline). Order: near-left, near-right, far-right, far-left.
COURT_IMG = np.float32([[210, 500], [750, 500], [610, 170], [350, 170]])
# Canonical top-down court (ITF singles: 23.77m x 8.23m), metres -> we keep px.
COURT_TOP = np.float32([[0, 0], [823, 0], [823, 2377], [0, 2377]])


def _perspective_point(u: float, v: float) -> np.ndarray:
    """Map court-relative (u,v) in [0,1]x[0,1] (u across, v along) to image px
    by bilinear interpolation of the trapezoid corners."""
    nl, nr, fr, fl = COURT_IMG
    near = nl + u * (nr - nl)
    far = fl + u * (fr - fl)
    return near + v * (far - near)


def _draw_disc(img, cx, cy, r, color):
    y0, y1 = max(0, int(cy - r)), min(H, int(cy + r + 1))
    x0, x1 = max(0, int(cx - r)), min(W, int(cx + r + 1))
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    img[y0:y1, x0:x1][mask] = color


def _draw_line(img, p0, p1, color, thick=2):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    n = int(max(abs(p1 - p0).max(), 1))
    for t in np.linspace(0, 1, n * 2):
        p = p0 + t * (p1 - p0)
        _draw_disc(img, p[0], p[1], thick, color)


def _draw_player(img, foot_px, scale, color, arm_ext):
    """Articulated stick figure at foot position. arm_ext in [-1,1] swings the
    hitting arm so pose/stroke phase is recoverable. Returns keypoints dict."""
    fx, fy = float(foot_px[0]), float(foot_px[1])
    h = 90 * scale
    hip = np.array([fx, fy - h * 0.5])
    shoulder = np.array([fx, fy - h * 0.95])
    head = np.array([fx, fy - h * 1.05])
    knee_l = np.array([fx - 8 * scale, fy - h * 0.22])
    knee_r = np.array([fx + 8 * scale, fy - h * 0.22])
    foot_l = np.array([fx - 12 * scale, fy])
    foot_r = np.array([fx + 12 * scale, fy])
    hand = shoulder + np.array([28 * scale * arm_ext, -18 * scale * abs(arm_ext)])
    elbow = (shoulder + hand) / 2 + np.array([0, 6 * scale])
    off_hand = shoulder + np.array([-18 * scale, 4 * scale])
    _draw_line(img, hip, shoulder, color, int(3 * scale))          # torso
    _draw_line(img, hip, knee_l, color, int(2 * scale)); _draw_line(img, knee_l, foot_l, color, int(2 * scale))
    _draw_line(img, hip, knee_r, color, int(2 * scale)); _draw_line(img, knee_r, foot_r, color, int(2 * scale))
    _draw_line(img, shoulder, elbow, color, int(2 * scale)); _draw_line(img, elbow, hand, color, int(2 * scale))
    _draw_line(img, shoulder, off_hand, color, int(2 * scale))
    _draw_disc(img, head[0], head[1], 6 * scale, color)
    return {"head": head, "shoulder": shoulder, "hip": hip, "hand": hand,
            "elbow": elbow, "knee_l": knee_l, "knee_r": knee_r,
            "foot_l": foot_l, "foot_r": foot_r}


def _court_background():
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (60, 90, 55)               # grass-ish surround (BGR-ish, we stay RGB)
    # fill court polygon
    poly = COURT_IMG.astype(np.int32)
    import cv2
    cv2.fillConvexPoly(img, poly, (150, 110, 75))     # clay-ish court
    # lines
    white = (235, 235, 235)
    nl, nr, fr, fl = COURT_IMG
    _draw_line(img, nl, nr, white, 2); _draw_line(img, fl, fr, white, 2)
    _draw_line(img, nl, fl, white, 2); _draw_line(img, nr, fr, white, 2)
    # net (mid, v=0.5)
    _draw_line(img, _perspective_point(0, 0.5), _perspective_point(1, 0.5), (245, 245, 245), 2)
    # centre service line
    _draw_line(img, _perspective_point(0.5, 0.25), _perspective_point(0.5, 0.75), white, 1)
    return img


def render_rally(seed: int = 7, n_shots: int = 6):
    """Render a rally. Returns (frames, truth). Players trade shots across the
    net; the ball follows a parabola with a bounce between hits."""
    rng = np.random.default_rng(seed)
    bg = _court_background()

    # Players start near their baselines, centre-ish.
    p_side = [0.90, 0.10]               # along-court home row (near / far baseline)
    shots = []                          # (hit_frame, hitter, from_uv, to_uv)
    frame_ball = []                     # per-frame (u, v, height) ; height fakes arc
    hit_frames = []
    hitter_seq = []

    hitter = 0
    cur_uv = np.array([0.5, p_side[0]])
    frames_per_shot = 22
    for s in range(n_shots):
        tgt_u = float(np.clip(rng.normal(0.5, 0.28), 0.05, 0.95))
        tgt_v = p_side[1 - hitter] + (0.06 if hitter == 0 else -0.06)
        tgt = np.array([tgt_u, tgt_v])
        hit_frames.append(len(frame_ball))
        hitter_seq.append(hitter)
        shots.append((len(frame_ball), hitter, cur_uv.copy(), tgt.copy()))
        for f in range(frames_per_shot):
            t = f / (frames_per_shot - 1)
            uv = cur_uv + t * (tgt - cur_uv)
            height = np.sin(t * np.pi) * 0.9          # arc; bounce ~ mid
            frame_ball.append((uv[0], uv[1], height))
        cur_uv = tgt
        hitter = 1 - hitter

    # Per-frame player positions: a player travels to meet each ball they hit,
    # then recovers toward centre. Continuous motion makes them foreground for
    # detection AND makes court-coverage a real number.
    n = len(frame_ball)
    kf = {0: [(0, 0.5, p_side[0])], 1: [(0, 0.5, p_side[1])]}
    for (hf, who, from_uv, _to) in shots:
        kf[who].append((hf, float(from_uv[0]), float(from_uv[1])))          # arrive to hit
        kf[who].append((min(n - 1, hf + 8), 0.5, p_side[who]))             # recover
    for pl in (0, 1):
        kf[pl].append((n - 1, 0.5, p_side[pl]))
        kf[pl].sort()
    player_uv = [[None, None] for _ in range(n)]
    for pl in (0, 1):
        ks = kf[pl]
        for a in range(len(ks) - 1):
            f0, u0, v0 = ks[a]; f1, u1, v1 = ks[a + 1]
            for f in range(f0, f1 + 1):
                t = (f - f0) / max(1, f1 - f0)
                player_uv[f][pl] = (u0 + t * (u1 - u0), v0 + t * (v1 - v0))

    frames = []
    truth = {"ball": [], "players": [], "court_img": COURT_IMG.tolist(),
             "hits": hit_frames, "hitters": hitter_seq, "fps": FPS, "wh": [W, H]}
    for i in range(n):
        img = bg.copy()
        u, v, hgt = frame_ball[i]
        # scale players by court depth (v): near (v~1) bigger, far (v~0) smaller
        (u0, v0), (u1, v1) = player_uv[i][0], player_uv[i][1]
        pos0 = _perspective_point(u0, v0); sc0 = 0.7 + 0.6 * v0
        pos1 = _perspective_point(u1, v1); sc1 = 0.7 + 0.6 * v1
        # swing arm near a hit frame
        def arm_for(pl):
            near = min((abs(i - hf), hk) for hf, hk in zip(hit_frames, hitter_seq))
            d, who = near
            return (1.0 - min(d, 6) / 6.0) if who == pl and d <= 6 else 0.15
        kp0 = _draw_player(img, pos0, sc0, (40, 40, 230), arm_for(0))     # player 0 red
        kp1 = _draw_player(img, pos1, sc1, (230, 210, 40), arm_for(1))    # player 1 yellow-ish
        # ball: lift by height (toward top of frame) and scale by depth
        ball_px = _perspective_point(u, v).copy()
        ball_px[1] -= hgt * 55 * (0.6 + 0.6 * v)
        br = max(2, int(4 * (0.6 + 0.7 * v)))
        _draw_disc(img, ball_px[0], ball_px[1], br, (60, 240, 240))       # bright yellow ball
        frames.append(img)
        truth["ball"].append([float(ball_px[0]), float(ball_px[1]), float(hgt)])
        truth["players"].append({
            "0": {"foot": pos0.tolist(), "kp": {k: v2.tolist() for k, v2 in kp0.items()}},
            "1": {"foot": pos1.tolist(), "kp": {k: v2.tolist() for k, v2 in kp1.items()}},
        })
    return frames, truth
