"""The README's model metrics table must match the recorded evaluation run
(docs/metrics.json, written by command_center/tools/evaluate_models.py) - no hand-edited or
estimated numbers, and unmeasured models say so."""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load():
    with open(os.path.join(ROOT, "docs", "metrics.json"), encoding="utf-8") as fh:
        metrics = json.load(fh)
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    section = readme.split("## Model metrics", 1)[1].split("\n## ", 1)[0]
    return metrics, section


ROW_LABELS = {"fire_smoke": "Fire/smoke (custom YOLOv8n)", "aerial_pose": "Aerial pose, SARD (custom YOLOv8n)",
              "tracking": "Object tracking", "fall_detection": "Fall detection"}


def rows(section, label):
    return [line for line in section.splitlines() if line.startswith("| " + label)]


def test_every_model_is_in_the_table():
    metrics, section = load()
    assert set(metrics["models"]) == set(ROW_LABELS)
    for key in metrics["models"]:
        assert rows(section, ROW_LABELS[key]), key


@pytest.mark.parametrize("key", ["fire_smoke", "aerial_pose"])
def test_measured_rows_match_the_evaluation_run(key):
    metrics, section = load()
    model = metrics["models"][key]
    assert model["status"] == "measured"
    table = rows(section, ROW_LABELS[key])
    assert len(table) == len(model["runs"])
    for run, line in zip(model["runs"], table, strict=True):
        cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
        assert cells[2] == f"{run['images']} / {run['instances']}"
        expected = [run[k] for k in ("precision", "recall", "f1", "mAP50", "mAP50_95")]
        assert [float(c) for c in cells[3:8]] == expected
        assert abs(run["f1"] - 2 * run["precision"] * run["recall"] / (run["precision"] + run["recall"])) < 0.002


@pytest.mark.parametrize("key", ["tracking", "fall_detection"])
def test_unmeasured_models_say_not_measured(key):
    metrics, section = load()
    assert metrics["models"][key]["status"] == "not measured"
    line = rows(section, ROW_LABELS[key])[0]
    assert line.count("not measured") == 5 and not any(ch.isdigit() for ch in line.split("|", 4)[-1])


def test_the_aerial_recall_caveat_is_documented():
    _, section = load()
    assert "Aerial recall of 0.512" in section and "human confirmation" in section
