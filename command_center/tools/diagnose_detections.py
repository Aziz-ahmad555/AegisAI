"""
Raw YOLO detection diagnostic - no tracker, low confidence threshold.

Answers "is the model seeing object X at all, and how confidently?" and
"what does each model / input size cost in latency on this machine?".

Captures frames from the webcam once, then runs every requested
model x input-size combination over the *same* frames, so results are
directly comparable.

Usage (from command_center/, with the app stopped so the camera is free):
    python tools/diagnose_detections.py                       # 10 s capture, yolov8n @ 320/416/640
    python tools/diagnose_detections.py --seconds 15 --models yolov8n.pt yolov8s.pt
    python tools/diagnose_detections.py --frames saved_frames/  # reuse a folder of images

Models that aren't present locally are skipped (never auto-downloaded)
unless --allow-download is passed.
"""
import argparse
import glob
import os
import statistics
import sys
import time
from collections import defaultdict

import cv2

DARK_MEAN = 25  # mean pixel value below which a frame is effectively black
WARMUP_TIMEOUT = 10.0  # seconds to wait for the camera to deliver lit frames
WATCH = {"cell phone", "person", "remote", "book", "laptop"}


def capture(seconds, every):
    # Open the camera through the app's own function so this capture uses
    # the exact same index, backend and resolution as Live Vision.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vision_stream import open_camera

    cap = open_camera()
    if not cap.isOpened():
        sys.exit("No camera available (is the Command Center still running? it holds the camera)")

    # Warm-up: some webcams deliver black frames for several seconds after
    # opening (measured ~5 s on the dev laptop). Waiting for brightness to be
    # *stable* isn't enough - a black stream is perfectly stable - so wait
    # until frames are actually lit, then for exposure to settle.
    t0, prev, stable, b = time.time(), None, 0, 0.0
    while time.time() - t0 < WARMUP_TIMEOUT and stable < 5:
        ok, frame = cap.read()
        if not ok:
            continue
        b = frame.mean()
        lit = b >= DARK_MEAN
        stable = stable + 1 if lit and prev is not None and abs(b - prev) < 2.0 else 0
        prev = b
    if b < DARK_MEAN:
        print(f"WARNING: camera still dark after {WARMUP_TIMEOUT:.0f}s (brightness {b:.1f}/255) - "
              "lens covered or room too dark?")
    else:
        print(f"Camera warmed up in {time.time() - t0:.1f}s, brightness {b:.1f}/255")

    print(f"Capturing for {seconds}s - hold the object up to the camera NOW...")
    frames, n, t0 = [], 0, time.time()
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if ok and n % every == 0:
            frames.append(frame)
        n += ok
    cap.release()
    return frames


def load_folder(path):
    files = sorted(glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.jpg")))
    return [cv2.imread(f) for f in files]


def evaluate(model_path, imgsz, frames, conf, out_dir):
    from ultralytics import YOLO

    model = YOLO(model_path)
    model.predict(frames[0], imgsz=imgsz, verbose=False)  # warm-up, not timed

    per_class = defaultdict(list)  # class -> confidences (best per frame)
    times = []
    best = (0.0, None)
    for frame in frames:
        t = time.perf_counter()
        r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        times.append((time.perf_counter() - t) * 1000)
        frame_best = {}
        for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name = r.names[int(c)]
            frame_best[name] = max(s, frame_best.get(name, 0.0))
        for name, s in frame_best.items():
            per_class[name].append(s)
        phone = frame_best.get("cell phone", 0.0)
        if phone > best[0]:
            best = (phone, r.plot())

    tag = f"{os.path.splitext(os.path.basename(model_path))[0]}@{imgsz}"
    ms = statistics.median(times)
    print(f"\n=== {tag}  (conf >= {conf}, no tracker, {len(frames)} frames) ===")
    print(f"  inference: median {ms:.0f} ms, p90 {sorted(times)[int(len(times) * 0.9) - 1]:.0f} ms  "
          f"-> ~{1000 / ms:.1f} fps ceiling")
    if not per_class:
        print("  no detections at all")
    for name, confs in sorted(per_class.items(), key=lambda kv: -len(kv[1])):
        mark = " <--" if name in WATCH else ""
        print(f"  {name:14s} in {len(confs):3d}/{len(frames)} frames  "
              f"max {max(confs):.2f}  median {statistics.median(confs):.2f}{mark}")
    if "cell phone" not in per_class:
        print("  cell phone: NOT detected in any frame")
    if best[1] is not None and out_dir:
        path = os.path.join(out_dir, f"best_phone_{tag}.jpg")
        cv2.imwrite(path, best[1])
        print(f"  best cell-phone frame saved: {path}")
    return per_class, ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--every", type=int, default=3, help="keep every Nth captured frame")
    ap.add_argument("--frames", help="folder of images to use instead of the webcam")
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[320, 416, 640])
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--out", default="diagnostics")
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args()

    frames = load_folder(args.frames) if args.frames else capture(args.seconds, args.every)
    frames = [f for f in frames if f is not None]
    if not frames:
        sys.exit("No frames.")

    levels = [f.mean() for f in frames]
    print(f"{len(frames)} frames, brightness mean {statistics.mean(levels):.1f} "
          f"min {min(levels):.1f} max {max(levels):.1f} (/255)")
    # Black frames can't contain detections and would skew hit rates, so they
    # are excluded from the analysis (and reported). All raw frames are saved.
    lit = [f for f, b in zip(frames, levels) if b >= DARK_MEAN]
    if len(lit) < len(frames):
        print(f"Excluding {len(frames) - len(lit)} black frame(s) (brightness < {DARK_MEAN}); analysing {len(lit)}.")
    if not lit:
        sys.exit("All frames are black - camera covered, privacy shutter closed, or the room is dark.")

    os.makedirs(args.out, exist_ok=True)
    if args.frames is None:
        raw_dir = os.path.join(args.out, "frames")
        os.makedirs(raw_dir, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(raw_dir, f"frame_{i:03d}.png"), f)
        print(f"Raw frames saved to {raw_dir}/ (reuse with --frames {raw_dir})")

    for model_path in args.models:
        if not os.path.exists(model_path) and not args.allow_download:
            print(f"\n=== {model_path}: not found locally, skipped (pass --allow-download to fetch it) ===")
            continue
        for size in args.sizes:
            evaluate(model_path, size, lit, args.conf, args.out)


if __name__ == "__main__":
    main()
