"""End-to-end tennis-vision pipeline: perception -> modeling -> application.

    python -m tennis_vision.pipeline                 # synthetic rally (default, self-contained)
    python -m tennis_vision.pipeline --seed 12       # a different synthetic rally
    python -m tennis_vision.pipeline --video clip.mp4  # a real video file

Writes to tennis_vision/out/:  annotated.mp4, dashboard.html, report.json
On the synthetic path it also VALIDATES perception against ground truth (mean
ball-position error in px, hit-detection recall, court-corner error) and prints
the numbers — because a prototype that can't say how wrong it is isn't finished.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import synth, perception, modeling, application

OUT = Path(__file__).resolve().parent / "out"


def _load_video(path: str):
    import imageio
    rdr = imageio.get_reader(path)
    fps = int(rdr.get_meta_data().get("fps", 30))
    frames = [np.asarray(f)[..., :3] for f in rdr]
    return frames, fps


def validate(perc: dict, model: dict, truth: dict) -> dict:
    """Score perception against synthetic ground truth."""
    # ball position error over frames where both exist
    errs = []
    for det, gt in zip(perc["ball"], truth["ball"]):
        if det["xy"] and not det.get("interp"):
            errs.append(np.hypot(det["xy"][0] - gt[0], det["xy"][1] - gt[1]))
    ball_err = float(np.mean(errs)) if errs else None
    ball_detect_rate = float(np.mean([1.0 if b["xy"] else 0.0 for b in perc["ball"]]))

    # hit recall: detected hit within +/-4 frames of a true hit
    true_hits = set(truth["hits"])
    det_frames = [h["frame"] for h in model["hits"]]
    matched = sum(any(abs(df - th) <= 4 for df in det_frames) for th in true_hits)
    recall = matched / len(true_hits) if true_hits else 0.0

    # court corner error (px), best-matched ordering
    ce = None
    if perc["court"]["corners_img"]:
        gt_c = np.array(truth["court_img"]); det_c = np.array(perc["court"]["corners_img"])
        ce = float(np.mean([np.min(np.hypot(*(gt_c - d).T)) for d in det_c]))

    return {
        "ball_pos_error_px_mean": None if ball_err is None else round(ball_err, 2),
        "ball_detection_rate": round(ball_detect_rate, 3),
        "hit_recall": round(recall, 3),
        "hits_detected_vs_true": f"{len(det_frames)} / {len(true_hits)}",
        "court_corner_error_px": None if ce is None else round(ce, 1),
    }


def run(video: str | None = None, seed: int = 7) -> dict:
    OUT.mkdir(exist_ok=True)
    truth = None
    if video:
        print(f"[input] decoding real video {video}")
        frames, fps = _load_video(video)
    else:
        print(f"[input] rendering synthetic rally (seed={seed})")
        frames, truth = synth.render_rally(seed=seed)
        fps = truth["fps"]
    print(f"[input] {len(frames)} frames @ {fps}fps")

    print("[layer 1] perception: court + ball + players")
    perc = perception.run_perception(frames)
    print(f"          court conf={perc['court']['conf']}  "
          f"ball frames={sum(1 for b in perc['ball'] if b['xy'])}/{len(frames)}")

    print("[layer 2] modeling: hits + strokes + stats")
    model = modeling.run_modeling(perc, fps)
    print(f"          {model['stats']['n_shots']} shots  "
          f"max {model['stats']['ball_speed_kmh']['max']} km/h  "
          f"overall conf {model['stats']['confidence']['overall']}")

    valid = validate(perc, model, truth) if truth else None
    if valid:
        print("[validate] vs ground truth:")
        for k, v in valid.items():
            print(f"           {k:28} {v}")

    print("[layer 3] application: annotated video + dashboard")
    ann = application.annotate(frames, perc, model)
    vid_path = application.write_video(ann, str(OUT / "annotated.mp4"), fps)
    dash_path = application.build_dashboard(perc, model, valid, str(OUT / "dashboard.html"))

    report = {"fps": fps, "n_frames": len(frames), "stats": model["stats"],
              "match_state": model["match_state"],
              "strokes": [{k: s[k] for k in ("frame", "hitter_id", "stroke", "conf")}
                          for s in model["strokes"]],
              "validation": valid}
    # heatmap point clouds bloat the json; drop them
    report["stats"] = {k: v for k, v in report["stats"].items() if k != "heatmap"}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\n[done] {vid_path}\n       {dash_path}\n       {OUT/'report.json'}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Tennis vision prototype pipeline")
    ap.add_argument("--video", help="path to a real video file (else synthetic)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    run(video=args.video, seed=args.seed)


if __name__ == "__main__":
    main()
