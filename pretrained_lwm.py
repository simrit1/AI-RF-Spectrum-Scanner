from __future__ import annotations

"""Pretrained LWM-Spectro classifier with upstream compatibility fixes.

The public model repository currently documents MoEPredictor inference, but its
main model snapshot omits mixture/train_top1_router.py even though
train_embedding_router.py imports it. This loader intentionally separates:

1) model artifacts from the public *model* repo (MoE + three experts), and
2) compatible Python inference code from the authors' historical *Space*
   revision where train_embedding_router.py and train_top1_router.py coexist.

No user training is required.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import sys
import numpy as np


@dataclass
class PretrainedPrediction:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    source: str


class LWMSpectroClassifier:
    REPO_ID = "wi-lab/lwm-spectro"
    # Pin known-compatible public revisions instead of silently depending on
    # whatever happens to be on main in the future.
    MODEL_REVISION = "79818e08babaf7b1a58e48bed1a01aba316dc49c"
    SPACE_CODE_REVISION = "24c4d80cd54a9ab69a1c5c4334eda37c7f844161"

    def __init__(self, cache_dir: Optional[str] = None, device: Optional[str] = None):
        self.cache_dir = cache_dir
        self.device_name = device
        self.predictor = None
        self.torch = None
        self.code_repo_dir = None
        self.artifact_repo_dir = None
        self.source = "pretrained:LWM-Spectro/wi-lab"
        self.warning = (
            "Public pretrained LTE/WiFi/5G LWM-Spectro model. The scanner uses "
            "a pinned author-published compatibility code revision because the "
            "current model snapshot omits one imported Python file."
        )

    @staticmethod
    def _expert_filename(comm: str) -> str:
        c = str(comm).strip().lower()
        if c == "wifi":
            return "WiFi_expert.pth"
        if c == "lte":
            return "LTE_expert.pth"
        if c == "5g":
            return "5G_expert.pth"
        raise ValueError(f"Unsupported LWM communication expert: {comm}")

    def _load(self) -> None:
        if self.predictor is not None:
            return
        try:
            import torch
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Pretrained LWM mode needs torch and huggingface_hub. "
                "Install with: python -m pip install -r requirements-pretrained.txt"
            ) from exc

        # Download only the actual model artifacts (about 62 MB total), not the
        # two ~383 MB demo datasets.
        artifact_repo = snapshot_download(
            repo_id=self.REPO_ID,
            revision=self.MODEL_REVISION,
            cache_dir=self.cache_dir,
            allow_patterns=[
                "moe_checkpoint.pth",
                "experts/LTE_expert.pth",
                "experts/WiFi_expert.pth",
                "experts/5G_expert.pth",
                "config.json",
            ],
        )

        # The current model repo is missing mixture/train_top1_router.py. Pull
        # the authors' own earlier Space revision where that dependency and the
        # matching train_embedding_router.py are both present.
        code_repo = snapshot_download(
            repo_id=self.REPO_ID,
            repo_type="space",
            revision=self.SPACE_CODE_REVISION,
            cache_dir=self.cache_dir,
            allow_patterns=["*.py", "**/*.py"],
        )

        self.code_repo_dir = code_repo
        self.artifact_repo_dir = artifact_repo

        # Make sure a prior failed import from the incomplete model snapshot
        # cannot poison this process.
        for name in list(sys.modules):
            if (
                name == "mixture"
                or name.startswith("mixture.")
                or name == "pretraining"
                or name.startswith("pretraining.")
                or name == "task1"
                or name.startswith("task1.")
                or name == "task2"
                or name.startswith("task2.")
            ):
                sys.modules.pop(name, None)

        if code_repo in sys.path:
            sys.path.remove(code_repo)
        sys.path.insert(0, code_repo)

        try:
            import mixture.train_embedding_router as embedding_router
        except Exception as exc:
            raise RuntimeError(
                "LWM-Spectro compatibility code could not be imported. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        # The released MoE checkpoint may contain training-machine paths for the
        # experts. Replace that path-resolution step with the three public expert
        # files downloaded from the model repo. This does not alter weights.
        def _public_expert_specs(checkpoint):
            specs = []
            entries = checkpoint.get("experts", [])
            if not entries:
                raise RuntimeError("LWM checkpoint contains no expert metadata")
            for expert in entries:
                comm = embedding_router.canonical_comm_name(str(expert["comm"]))
                expert_path = Path(artifact_repo) / "experts" / self._expert_filename(comm)
                if not expert_path.exists():
                    raise FileNotFoundError(f"Public LWM expert checkpoint missing: {expert_path}")
                specs.append(
                    embedding_router.ExpertSpec(
                        name=str(expert.get("name", f"{comm}_expert")),
                        comm=comm,
                        checkpoint=expert_path,
                        stats_path=None,
                    )
                )
            return specs

        embedding_router._checkpoint_to_expert_specs = _public_expert_specs
        MoEPredictor = embedding_router.MoEPredictor

        ckpt = Path(artifact_repo) / "moe_checkpoint.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"LWM MoE checkpoint missing: {ckpt}")

        device = torch.device(self.device_name) if self.device_name else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        try:
            self.predictor = MoEPredictor.from_checkpoint(ckpt, device=device)
        except Exception as exc:
            raise RuntimeError(
                "LWM-Spectro model files were downloaded, but the public MoE could "
                "not be constructed. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        self.torch = torch

    @staticmethod
    def _technology_from_routing(routing) -> tuple[str, float]:
        if not routing:
            return "Unknown", 0.0
        best = max(routing, key=lambda r: float(r.get("weight", 0.0)))
        comm = str(best.get("comm", "Unknown"))
        return comm, float(best.get("weight", 0.0))

    def predict(
        self,
        iq: np.ndarray,
        sample_rate: float,
        component_offset_hz: float,
        component_bandwidth_hz: float,
        force_wideband: bool = False,
    ) -> PretrainedPrediction:
        if float(component_bandwidth_hz) < 3.0e6 and not force_wideband:
            return PretrainedPrediction("not_applicable", 0.0, {}, self.source)

        self._load()
        from cnn_rf import iq_to_spectrogram

        if force_wideband:
            spec_offset = 0.0
            spec_bw = None
        else:
            spec_offset = component_offset_hz
            spec_bw = component_bandwidth_hz

        img = iq_to_spectrogram(
            iq,
            sample_rate,
            component_offset_hz=spec_offset,
            component_bandwidth_hz=spec_bw,
            image_size=128,
        )
        spec = self.torch.from_numpy(img.astype(np.float32))
        out = self.predictor.predict(spec, return_probabilities=True, return_routing=True)

        raw_label = str(out.get("label", out.get("predicted_class", "Unknown")))
        conf = float(out.get("confidence", 0.0))

        label_norm = raw_label.strip().lower()
        if label_norm not in {"wifi", "lte", "5g"}:
            route_label, route_conf = self._technology_from_routing(out.get("routing"))
            if route_label.strip().lower() in {"wifi", "lte", "5g"}:
                raw_label = route_label
                conf = max(conf, route_conf)

        probs = out.get("probabilities", {})
        if isinstance(probs, list):
            probs = {f"class_{i}": float(v) for i, v in enumerate(probs)}
        elif not isinstance(probs, dict):
            probs = {}

        return PretrainedPrediction(raw_label, conf, probs, self.source)
