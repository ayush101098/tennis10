# Tennis Vision — functioning prototype

A runnable, end-to-end slice of the PRD's three-layer architecture
(Perception → Modeling → Application). It processes a tennis rally video and
produces an annotated overlay video, a self-contained HTML dashboard, and a JSON
report — with per-output **confidence** and, on the synthetic input, **accuracy
measured against ground truth** (the PRD's core validation principle).

## Run

```bash
# self-contained default: renders a synthetic rally WITH ground truth, then
# runs all three layers and validates perception against the labels
python -m tennis_vision.pipeline
python -m tennis_vision.pipeline --seed 12      # a different rally

# a real video file (fixed-camera clip works best — see limits below)
python -m tennis_vision.pipeline --video path/to/clip.mp4
```

Outputs land in `tennis_vision/out/`: `annotated.mp4`, `dashboard.html`
(open in any browser — everything is inlined), `report.json`.

## How the code maps to the PRD

| PRD layer | Module | What it does now |
|---|---|---|
| Layer 0 (validation input) | [`synth.py`](synth.py) | Renders a physically-plausible rally in broadcast perspective and emits exact ball/player/court/hit labels, so accuracy is *measurable*, not asserted |
| **Layer 1 — Perception** | [`perception.py`](perception.py) | Court homography → metric court mm; ball tracking (colour blob + gap-filling, per-frame confidence); player detection & ID tracking (background subtraction) |
| **Layer 2 — Modeling** | [`modeling.py`](modeling.py) | Hit detection, stroke labels (serve/FH/BH), rally stats (ball km/h, per-player distance run, coverage heatmap), toy match-state — all confidence-tagged |
| **Layer 3 — Application** | [`application.py`](application.py) | Annotated overlay video + one-file HTML dashboard (trajectory, speed, coverage charts, shot table, confidence panel) |
| Orchestration + validation | [`pipeline.py`](pipeline.py) | Wires the layers; scores perception vs ground truth on the synthetic path |

## Latest validated run (synthetic, seed 7)

```
ball_pos_error_px_mean   0.18      ball_detection_rate  1.00
hit_recall               0.833     court_corner_error   3.9 px
```

## Honest limits (these are the real-video swap points, called out per the PRD)

- **Ball detection** is a bright-blob detector. Robust on controlled/clean input;
  a real broadcast ball needs a trained small-object model (e.g. TrackNet).
  Swap point: `perception.detect_ball()`.
- **Player detection** is fixed-camera background subtraction. Broadcast cuts /
  moving cameras need a learned detector (YOLO) + re-ID. Swap: `detect_players()`.
- **Pose / technique** is not yet a trained model — stroke labels are heuristic.
  Real technique scoring needs a pose model on labelled strokes (the PRD's P0
  "calibrate against certified coaches" item). Function signatures already leave
  room for keypoints.
- **Match-state** is a toy cadence indicator, deliberately labelled as such — not
  a validated win-probability model.

Accuracy depends entirely on input configuration; the dashboard states which mode
each run used, matching the PRD requirement to report accuracy per configuration
rather than as a single blanket number.
