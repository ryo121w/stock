"""Model store for versioning and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from qtp.models.base import ModelWrapper
from qtp.models.lgbm import LGBMPipeline

logger = structlog.get_logger()

MODEL_CLASSES = {
    "lgbm": LGBMPipeline,
}


class ModelStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: ModelWrapper, metrics: dict | None = None) -> str:
        model_dir = self.base_dir / model.version
        model.save(model_dir)
        if metrics:
            (model_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
        return model.version

    def load(self, version: str) -> ModelWrapper:
        model_dir = self.base_dir / version
        model_type = version.split("_")[0]
        cls = MODEL_CLASSES.get(model_type, LGBMPipeline)
        return cls.load(model_dir)

    def load_latest(self) -> ModelWrapper:
        versions = self.list_versions()
        if not versions:
            raise FileNotFoundError("No models found")
        latest = versions[-1]
        return self.load(latest["version"])

    def list_versions(self) -> list[dict]:
        versions = []
        for d in sorted(self.base_dir.iterdir()):
            if d.is_dir() and (d / "metadata.json").exists():
                meta = json.loads((d / "metadata.json").read_text())
                versions.append({"version": meta["version"], "created_at": meta.get("created_at")})
        return versions
