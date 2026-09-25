"""Reproduce the detection metrics published in the README.

Runs Ultralytics validation for every model that has local weights AND a
labelled dataset, and writes the raw results to docs/metrics.json. Models
without a labelled evaluation set are recorded as "not measured" - no number
is ever estimated or copied from elsewhere.

    command_center\\venv\\Scripts\\python.exe command_center\\tools\\evaluate_models.py

CPU only; takes a few minutes. Datasets and weights are git-ignored (see
archive/phases/README.md for how to download/retrain them).
"""
import json
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone

import ultralytics
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASES = os.path.join(ROOT, "archive", "phases")
OUT = os.path.join(ROOT, "docs", "metrics.json")

FIRE_WEIGHTS = os.path.join(ROOT, "command_center", "fire_smoke_model.pt")
FIRE_ZIP = os.path.join(PHASES, "phase3_fire_smoke", "FIRE-n-SMOKE-DETECTION-1", "roboflow.zip")
AERIAL_DIR = os.path.join(PHASES, "phase7_drone_intelligence")
AERIAL_WEIGHTS = os.path.join(AERIAL_DIR, "runs", "detect", "train", "weights", "best.pt")
AERIAL_DATA = os.path.join(AERIAL_DIR, "Sard-4")
AERIAL_TRAINING_LOG = os.path.join(AERIAL_DIR, "runs", "detect", "train", "results.csv")
IMGSZ = 416          # both models were trained at 416


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def validate(weights, data_yaml, split):
    """One Ultralytics validation run -> overall + per-class metrics."""
    metrics = YOLO(weights).val(data=data_yaml, split=split, imgsz=IMGSZ, batch=8, device="cpu",
                                plots=False, verbose=False, project=tempfile.mkdtemp(), name="val")
    box = metrics.box
    p, r = float(box.mp), float(box.mr)
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        cp, cr = float(box.p[i]), float(box.r[i])
        per_class[metrics.names[int(c)]] = {
            "precision": round(cp, 3), "recall": round(cr, 3), "f1": round(f1(cp, cr), 3),
            "mAP50": round(float(box.ap50[i]), 3), "mAP50_95": round(float(box.ap[i]), 3),
        }
    nt = getattr(metrics, "nt_per_class", None)
    return {
        "split": split,
        "images": None,                                  # filled in by the caller
        "instances": int(nt.sum()) if nt is not None else None,
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1(p, r), 3),
        "mAP50": round(float(box.map50), 3), "mAP50_95": round(float(box.map), 3),
        "per_class": per_class,
    }


def count_images(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def _extract_eval_splits(zip_path, dest):
    """Extract valid/ and test/ only, renaming files to short stems: Roboflow's
    filenames exceed the Windows 260-character path limit. An image and its
    label share a stem, so pairs stay matched."""
    stems = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) != 3 or parts[0] not in ("valid", "test") or not parts[2]:
                continue
            split, kind, filename = parts
            stem, ext = os.path.splitext(filename)
            short = stems.setdefault((split, stem), f"{len(stems):05d}")
            os.makedirs(os.path.join(dest, split, kind), exist_ok=True)
            with z.open(name) as src, open(os.path.join(dest, split, kind, short + ext), "wb") as out:
                shutil.copyfileobj(src, out)


def fire_smoke():
    if not (os.path.exists(FIRE_WEIGHTS) and os.path.exists(FIRE_ZIP)):
        return {"status": "not measured", "reason": "weights or dataset zip not present locally"}
    # The unpacked phase-3 folder is incomplete (no valid/ split, no train
    # labels); the original Roboflow export zip has every split.
    work = tempfile.mkdtemp(prefix="aegis-fire-")
    try:
        _extract_eval_splits(FIRE_ZIP, work)
        data = os.path.join(work, "data.yaml")
        with open(data, "w", encoding="utf-8") as fh:
            # train/ isn't extracted (not needed to validate); point it at valid/.
            fh.write(f"path: {work}\ntrain: valid/images\nval: valid/images\ntest: test/images\n"
                     "nc: 2\nnames: ['Fire', 'Smoke']\n")
        runs = []
        for split, folder in (("val", "valid"), ("test", "test")):
            run = validate(FIRE_WEIGHTS, data, split)
            run["images"] = count_images(os.path.join(work, folder, "images"))
            runs.append(run)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"status": "measured", "weights": "command_center/fire_smoke_model.pt",
            "dataset": "Roboflow 'FIRE n SMOKE DETECTION' v1 (CC BY 4.0)", "runs": runs}


def aerial():
    if not (os.path.exists(AERIAL_WEIGHTS) and os.path.isdir(AERIAL_DATA)):
        return {"status": "not measured", "reason": "weights or SARD dataset not present locally"}
    work = tempfile.mkdtemp(prefix="aegis-sard-")
    try:
        data = os.path.join(work, "data.yaml")
        names = ["Running", "Walking", "laying_down", "not_defined", "seated", "stands"]
        with open(data, "w", encoding="utf-8") as fh:
            fh.write(f"path: {AERIAL_DATA}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
                     f"nc: {len(names)}\nnames: {names}\n")
        runs = []
        for split, folder in (("val", "valid"), ("test", "test")):
            run = validate(AERIAL_WEIGHTS, data, split)
            run["images"] = count_images(os.path.join(AERIAL_DATA, folder, "images"))
            runs.append(run)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    with open(AERIAL_TRAINING_LOG, encoding="utf-8") as fh:
        last = fh.read().strip().splitlines()[-1].split(",")
    return {"status": "measured", "weights": "archive/phases/phase7_drone_intelligence/runs/detect/train/weights/best.pt",
            "dataset": "Roboflow SARD (Search and Rescue Drone) v4", "runs": runs,
            "training_log_final_epoch": {"epoch": int(last[0]), "precision": float(last[5]), "recall": float(last[6]),
                                         "mAP50": float(last[7]), "mAP50_95": float(last[8])}}


def main():
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"ultralytics": ultralytics.__version__, "python": sys.version.split()[0],
                        "platform": platform.platform(), "device": "cpu", "imgsz": IMGSZ},
        "models": {
            "fire_smoke": fire_smoke(),
            "aerial_pose": aerial(),
            "tracking": {"status": "not measured",
                         "reason": "stock COCO-pretrained YOLOv8n/s + ByteTrack; no labelled video in the repo "
                                   "to compute detection or tracking (MOTA/IDF1) metrics on"},
            "fall_detection": {"status": "not measured",
                               "reason": "rule on YOLOv8n-pose keypoints; no labelled fall/no-fall set in the repo"},
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
