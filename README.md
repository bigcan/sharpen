# FinRL Pro Scaffold

This repository hosts the FinRL Pro scaffolding that layers on top of the open-source FinRL Podracer project. The goal is to provide a clean extension surface so that future Spec-Kit deliverables can evolve FinRL without modifying the upstream FinRL_Podracer internals.

## Project Layout

- `finrl_pro/` – Python package exposing FinRL Pro modules (data, environments, agents, training, evaluation, MLOps, explainability, configs, utilities).
- `specs/` – Specification documents that will guide the implementation roadmap.
- `templates/` – Starter templates for reports, configuration files, or automation scripts.
- `tests/` – Pytest-based test suite placeholder validating scaffolding integrity.
- `notebooks/` – Jupyter notebooks for research and experimentation.
- `docker/` – Container definitions tailored to FinRL Pro deployments.
- `conf/` – Environment-specific configuration overlays.
- `.vscode/` – VS Code workspace defaults for a consistent developer experience.

## Getting Started

1. Create a local virtual environment and install in editable mode:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```
2. Run the smoke tests to confirm the scaffold imports:
   ```bash
   pytest
   ```
3. Add Spec-Kit driven implementations inside the placeholder modules under `finrl_pro/`.

## Next Steps

- Populate `specs/` with feature specifications.
- Replace placeholder classes and methods with production-ready implementations guided by specs.
- Wire MLOps, explainability, and evaluation modules to FinRL Podracer once interfaces are finalized.
