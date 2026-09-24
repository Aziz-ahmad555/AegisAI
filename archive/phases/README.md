# Archived phase snapshots

These folders are the **historical development snapshots** of AegisAI, one per build phase (Phase 1 vision through Phase 11 multi-agent / MLOps), kept exactly as they were when each phase was built and tested.

- They are **not maintained**. Code here is not kept in sync with the live system, and several modules are earlier copies of what now lives in `aegis_core`.
- The **live code** is:
  - [`aegis_core/`](../../aegis_core): the shared domain logic (digital twin, sensor fusion, anomaly/trend detection, NLP, agents, event bus, vision pipeline);
  - [`command_center/`](../../command_center): the web app that runs it.
- They stay in the repo because they document how the system was built. Each phase's run instructions are in the main [README](../../README.md#how-to-run-the-individual-phases-archived).
- Local-only content that moved with them (per-phase `venv/`, datasets, training `runs/`, model weights) is git-ignored. Windows venvs contain absolute paths, so a moved phase venv's `activate`/`pip.exe` may point at the old location. Recreate it or use `venv\Scripts\python.exe -m pip` if you need to run a phase again.
- Model training pipelines (to reproduce weights) live here: fire/smoke in `phase3_fire_smoke/`, aerial search-and-rescue in `phase7_drone_intelligence/`.
