# MLOps Practices in AegisAI

This document summarizes the model lifecycle practices followed throughout the project, consolidating what was applied incrementally across Phases 3 and 7 into a single reference.

## Model versioning

- Each custom-trained model (Phase 3 fire/smoke detector, Phase 7 aerial pose detector) is trained via a scripted, reproducible pipeline rather than trained interactively and manually saved.
- Training configuration (epochs, image size, batch size, base model) is recorded directly in the command used to invoke training, itself documented in this repository's README, rather than left implicit.
- Training run outputs are versioned by Ultralytics' own run-numbering (`runs/detect/train`, `train2`, etc.), preventing silent overwrites of prior results.

## Model registry (lightweight)

This project does not use a formal model registry (e.g., MLflow), since a single-developer CPU-only project doesn't yet justify that overhead. Instead:

- Trained weights are excluded from git (`.gitignore`) and instead reproducible from the dataset + training command documented per-phase in the README.
- Model performance metrics (precision, recall, mAP50, mAP50-95) are recorded in the README at the time of training, functioning as a lightweight, human-readable registry entry per model version.

| Model | Phase | Classes | mAP50 | Precision | Recall |
|---|---|---|---|---|---|
| Fire/Smoke Detector | 3 | Fire, Smoke | 0.576 | - | - |
| Aerial Pose Detector | 7 | Running, Walking, laying_down, seated, stands, not_defined | 0.572 | 0.784 | 0.512 |

## Monitoring (manual, not automated)

No automated model-monitoring service is deployed (would require production traffic this project doesn't have). Instead, monitoring was performed manually during development:

- **False positive tracking**: the fire/smoke model's false positives on skin/hands and low-light noise were identified through live webcam testing and documented as a known limitation rather than silently ignored.
- **Confidence threshold tuning**: default detection confidence was raised from 0.4 to 0.55 in Phase 5's integration after observing weak, low-confidence false triggers in practice - a manual equivalent of the confidence-distribution monitoring described in large-scale MLOps systems.

## Retraining triggers (documented, not automated)

Since this is a static portfolio project rather than a live production service, there is no automated retraining pipeline. The conditions that *would* trigger retraining in a production deployment are documented instead:

- Day/night domain gap in the fire/smoke model (Phase 3) would warrant retraining with a low-light-augmented dataset before any real deployment.
- The aerial pose detector's moderate recall (0.512) would benefit from a larger training set or more epochs before being trusted in an actual search-and-rescue context.

## Experiment tracking

Formal experiment tracking (e.g., Weights & Biases) was not used, since each model was trained once at a fixed configuration appropriate for the CPU hardware constraints of this project. Given more compute, a natural next step would be tracking multiple epoch/image-size/augmentation configurations against each other rather than a single fixed run per model.

## Why this scope, honestly

A full MLOps stack (model registry, automated retraining, drift detection, CI/CD for models) is designed for production systems with live traffic and a team maintaining them. Applying that full stack here would be performative rather than useful. What's documented above reflects the actual practices meaningfully applicable to a single-developer, CPU-only, portfolio-stage project - versioned reproducible training, honest performance recording, and manually-identified failure modes - while being explicit about what a production deployment of this system would still need to add.
