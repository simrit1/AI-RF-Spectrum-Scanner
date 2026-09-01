from __future__ import annotations

"""Spectrogram-CNN utilities for the Pluto house RF scanner.

The CNN sees normalized magnitude spectrograms only. It does not decode packet,
audio, identity, or payload content.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage, signal

CNN_CLASS_NAMES = [
    "broadcast_fm",
    "wifi_ofdm",
    "bluetooth_fhss",
    "zigbee_like",
    "narrowband",
    "bursty_ism",
    "chirp_css",
    "wideband_digital",
    "unknown_noise",
]

CNN_FRIENDLY = {
    "broadcast_fm": "Broadcast FM shape",
    "wifi_ofdm": "Wi-Fi/OFDM-like",
    "bluetooth_fhss": "Bluetooth/FHSS-like",
    "zigbee_like": "Zigbee/802.15.4-like",
    "narrowband": "Narrowband",
    "bursty_ism": "Bursty ISM/OOK/FSK-like",
    "chirp_css": "Chirp/CSS-like",
    "wideband_digital": "Wideband digital",
    "unknown_noise": "Unknown/noise",
}

DEFAULT_SPEC_SIZE = 96


def iq_to_spectrogram(
    iq: np.ndarray,
    sample_rate: float,
    component_offset_hz: float = 0.0,
    component_bandwidth_hz: Optional[float] = None,
    image_size: int = DEFAULT_SPEC_SIZE,
) -> np.ndarray:
    """Convert complex IQ to a robust normalized spectrogram image [0,1].

    Frequency is cropped around the detected component when possible. This makes
    the CNN focus on the signal morphology instead of the whole Pluto snapshot.
    """
    x = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if len(x) < 2048:
        raise ValueError("Need at least 2048 IQ samples for CNN spectrogram")

    # Normalize amplitude while preserving temporal structure.
    x = x - np.mean(x)
    scale = float(np.percentile(np.abs(x), 99.5))
    if scale > 0:
        x = x / scale

    nperseg = 512 if len(x) >= 8192 else 256
    noverlap = int(0.75 * nperseg)
    f, _t, z = signal.stft(
        x,
        fs=float(sample_rate),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    f = np.fft.fftshift(f)
    z = np.fft.fftshift(z, axes=0)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(z), 1e-8))

    if component_bandwidth_hz is not None and np.isfinite(component_bandwidth_hz):
        # Keep context around the component but avoid feeding the entire 20 MHz
        # snapshot for a 1 MHz signal.
        crop_bw = max(float(component_bandwidth_hz) * 2.8, float(sample_rate) * 0.08, 120e3)
        crop_bw = min(crop_bw, float(sample_rate) * 0.92)
        lo = float(component_offset_hz) - crop_bw / 2.0
        hi = float(component_offset_hz) + crop_bw / 2.0
        mask = (f >= lo) & (f <= hi)
        if int(np.count_nonzero(mask)) >= 12:
            mag_db = mag_db[mask, :]

    # Robust per-example normalization. Absolute power/SNR remains available to
    # the feature classifier and is intentionally not leaked into this image.
    p_lo, p_hi = np.percentile(mag_db, [5.0, 99.3])
    if p_hi <= p_lo + 1e-6:
        p_hi = p_lo + 1.0
    img = np.clip((mag_db - p_lo) / (p_hi - p_lo), 0.0, 1.0)

    zoom = (image_size / img.shape[0], image_size / img.shape[1])
    img = ndimage.zoom(img, zoom=zoom, order=1)
    img = img[:image_size, :image_size]
    if img.shape != (image_size, image_size):
        out = np.zeros((image_size, image_size), dtype=np.float32)
        out[: img.shape[0], : img.shape[1]] = img
        img = out
    return img.astype(np.float32)


def _torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for CNN mode. Install it with: pip install torch"
        ) from exc


def build_cnn(num_classes: int):
    torch, nn = _torch()

    class SpectrogramCNN(nn.Module):
        def __init__(self, nclasses: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 5, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 96, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.25),
                nn.Linear(96 * 4 * 4, 160),
                nn.ReLU(inplace=True),
                nn.Dropout(0.20),
                nn.Linear(160, nclasses),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return SpectrogramCNN(num_classes)


@dataclass
class CNNPrediction:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    source: str


class SpectrogramCNNClassifier:
    def __init__(self, model_path: str, device: str = "cpu"):
        torch, _nn = _torch()
        self.torch = torch
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = list(checkpoint["class_names"])
        self.image_size = int(checkpoint.get("image_size", DEFAULT_SPEC_SIZE))
        self.model = build_cnn(len(self.class_names)).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.source = str(checkpoint.get("source", "spectrogram-cnn"))
        self.warning = str(checkpoint.get("warning", ""))

    def predict(
        self,
        iq: np.ndarray,
        sample_rate: float,
        component_offset_hz: float,
        component_bandwidth_hz: float,
    ) -> CNNPrediction:
        img = iq_to_spectrogram(
            iq,
            sample_rate,
            component_offset_hz=component_offset_hz,
            component_bandwidth_hz=component_bandwidth_hz,
            image_size=self.image_size,
        )
        x = self.torch.from_numpy(img[None, None, :, :]).to(self.device)
        with self.torch.no_grad():
            logits = self.model(x)
            probs = self.torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return CNNPrediction(
            label=self.class_names[idx],
            confidence=float(probs[idx]),
            probabilities={k: float(v) for k, v in zip(self.class_names, probs)},
            source=self.source,
        )


def map_cnn_to_rf_label(cnn_label: str, freq_mhz: float, bandwidth_khz: float) -> str:
    raw = str(cnn_label).strip()
    low = raw.lower()
    # Public LWM-Spectro technology labels. Only map WiFi to WiFi when the
    # frequency/bandwidth are physically plausible; LTE/5G remain generic
    # wideband-digital evidence in this household scanner.
    if low == "wifi":
        if bandwidth_khz < 3000:
            return "unknown"
        if 2395 <= freq_mhz <= 2490:
            return "wifi_24"
        if 5150 <= freq_mhz <= 5900:
            return "wifi_5"
        return "wideband_digital"
    if low in {"lte", "5g"}:
        return "wideband_digital"
    if low in {"not_applicable", "unknown"}:
        return "unknown"
    if cnn_label == "broadcast_fm":
        return "broadcast_fm" if 87.5 <= freq_mhz <= 108.5 else "unknown"
    if cnn_label == "wifi_ofdm":
        if bandwidth_khz < 3000:
            return "unknown"
        if 2395 <= freq_mhz <= 2490:
            return "wifi_24"
        if 5150 <= freq_mhz <= 5900:
            return "wifi_5"
        return "wideband_digital"
    if cnn_label == "bluetooth_fhss":
        return "bluetooth_fhss" if 2395 <= freq_mhz <= 2490 else "narrowband"
    if cnn_label == "zigbee_like":
        return "zigbee_24" if 2395 <= freq_mhz <= 2490 else "wideband_digital"
    if cnn_label in {"bursty_ism", "chirp_css"}:
        if 431 <= freq_mhz <= 436:
            return "ism_433"
        if 901 <= freq_mhz <= 929:
            return "ism_915"
        if 312 <= freq_mhz <= 318 or 387 <= freq_mhz <= 393:
            return "garage_remote"
        return "narrowband" if bandwidth_khz < 500 else "wideband_digital"
    if cnn_label == "narrowband":
        return "narrowband"
    if cnn_label == "wideband_digital":
        return "wideband_digital"
    return "unknown"


def load_npz_example(path: str) -> Tuple[np.ndarray, float, float, float, str]:
    d = np.load(path, allow_pickle=False)
    iq = np.asarray(d["iq"], dtype=np.complex64)
    sr = float(d["sample_rate"])
    offset = float(d["component_offset_hz"]) if "component_offset_hz" in d else 0.0
    bw = float(d["component_bandwidth_hz"]) if "component_bandwidth_hz" in d else sr * 0.5
    label_raw = d["label"] if "label" in d else ""
    if isinstance(label_raw, np.ndarray):
        label_raw = label_raw.item()
    if isinstance(label_raw, bytes):
        label_raw = label_raw.decode("utf-8")
    return iq, sr, offset, bw, str(label_raw)
