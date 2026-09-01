from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal, stats

FEATURE_NAMES = [
    "freq_mhz",
    "bandwidth_khz",
    "snr_db",
    "spectral_flatness",
    "peakiness_db",
    "burstiness",
    "envelope_cv",
    "duty_cycle",
    "spectral_kurtosis",
]

FRIENDLY_LABELS = {
    "broadcast_fm": "Broadcast FM",
    "wifi_24": "Wi-Fi 2.4 GHz-like",
    "wifi_5": "Wi-Fi 5 GHz-like",
    "bluetooth_fhss": "Bluetooth/FHSS-like",
    "zigbee_24": "Zigbee / 802.15.4-like",
    "ism_433": "433 MHz ISM device-like",
    "ism_915": "902-928 MHz ISM / Z-Wave / LoRa-like",
    "garage_remote": "315/390 MHz remote-like",
    "narrowband": "Narrowband signal",
    "wideband_digital": "Wideband digital signal",
    "unknown": "Unknown",
}

@dataclass(frozen=True)
class ScanTarget:
    name: str
    center_hz: float
    sample_rate: float
    rf_bw: float
    dwell_s: float = 0.05
    min_snr_db: float = 8.0
    hint: str = ""

@dataclass
class Detection:
    timestamp: str
    scan_name: str
    center_freq_hz: float
    signal_freq_hz: float
    bandwidth_hz: float
    snr_db: float
    label: str
    friendly_label: str
    confidence: float
    spectral_flatness: float
    peakiness_db: float
    burstiness: float
    envelope_cv: float
    duty_cycle: float
    spectral_kurtosis: float
    model_source: str
    note: str = ""
    cnn_label: str = ""
    cnn_confidence: float = 0.0
    cnn_accepted: bool = True
    cnn_rejection_reason: str = ""
    feature_label: str = ""
    feature_confidence: float = 0.0

    def as_row(self) -> Dict[str, object]:
        return asdict(self)


def mhz(x: float) -> float:
    return x * 1e6


def build_scan_plan(profile: str = "house_quick", include_5g: bool = False) -> List[ScanTarget]:
    profile = profile.lower()
    plan: List[ScanTarget] = []

    # Broadcast FM: overlapping 2.4 MS/s chunks. This is intentionally lower-rate for reliable USB streaming.
    fm_centers = [89.2, 91.4, 93.6, 95.8, 98.0, 100.2, 102.4, 104.6, 106.8]
    for c in fm_centers:
        plan.append(ScanTarget("FM broadcast", mhz(c), 2.4e6, 2.2e6, 0.04, 9.0, "broadcast_fm"))

    # Common sub-GHz household / ISM regions in North America.
    for c in [315.0, 390.0]:
        plan.append(ScanTarget("Remote-control region", mhz(c), 2.4e6, 2.0e6, 0.06, 9.0, "garage_remote"))
    plan.append(ScanTarget("433 MHz ISM", mhz(433.92), 4.0e6, 3.0e6, 0.08, 8.0, "ism_433"))

    # 902-928 MHz. Quick mode uses 4 MHz slices; deep uses denser overlap.
    centers_915 = [904, 908, 912, 916, 920, 924, 927]
    if profile == "house_deep":
        centers_915 = list(np.arange(903, 928, 2.5))
    for c in centers_915:
        plan.append(ScanTarget("902-928 MHz ISM", mhz(float(c)), 4.0e6, 3.5e6, 0.08, 8.0, "ism_915"))

    # Wi-Fi snapshots. Short 20 MS/s buffers are used to characterize 20 MHz OFDM-like channels.
    # Centers cover the three non-overlapping 2.4 GHz Wi-Fi channels most commonly used in North America.
    for c in [2412, 2437, 2462]:
        plan.append(ScanTarget("2.4 GHz Wi-Fi snapshot", mhz(c), 20.0e6, 20.0e6, 0.025, 7.0, "wifi_24"))

    # Narrower 2.4 GHz sweep to expose BLE/Zigbee/FHSS activity that is easy to hide under a wide Wi-Fi capture.
    narrow_centers = list(np.arange(2404, 2482, 4.0))
    if profile == "house_quick":
        narrow_centers = [2404, 2412, 2420, 2428, 2436, 2444, 2452, 2460, 2468, 2476, 2480]
    for c in narrow_centers:
        plan.append(ScanTarget("2.4 GHz narrowband", mhz(float(c)), 4.0e6, 3.8e6, 0.035, 7.0, "bluetooth_fhss"))

    if include_5g:
        # Selected 20 MHz Wi-Fi channel centers. Extended Pluto tuning is user-dependent outside stock range.
        for c in [5180, 5200, 5220, 5240, 5745, 5765, 5785, 5805]:
            plan.append(ScanTarget("5 GHz Wi-Fi snapshot", mhz(c), 20.0e6, 20.0e6, 0.025, 7.0, "wifi_5"))

    if profile not in {"house_quick", "house_deep"}:
        raise ValueError("profile must be 'house_quick' or 'house_deep'")
    return plan


def build_custom_plan(start_hz: float, stop_hz: float, step_hz: float, sample_rate: float, rf_bw: Optional[float] = None) -> List[ScanTarget]:
    if start_hz <= 0 or stop_hz <= start_hz or step_hz <= 0 or sample_rate <= 0:
        raise ValueError("Invalid custom sweep settings")
    rf_bw = float(rf_bw or min(sample_rate, 20e6))
    return [
        ScanTarget("Custom sweep", float(fc), float(sample_rate), rf_bw, 0.05, 8.0, "")
        for fc in np.arange(start_hz, stop_hz + step_hz / 2.0, step_hz)
    ]


class PlutoReceiver:
    def __init__(self, uri: str = "ip:192.168.2.1", gain_mode: str = "slow_attack", manual_gain_db: float = 50.0):
        self.uri = uri
        self.gain_mode = gain_mode
        self.manual_gain_db = manual_gain_db
        self.sdr = None
        self._last_sr = None
        self._last_bw = None

    def connect(self) -> None:
        # pyadi-iio sits on top of libiio. On Windows it is possible to have
        # pyadi-iio installed while the native libiio DLL is missing from the
        # Python process. In that situation pylibiio/ctypes can raise the very
        # unhelpful: TypeError: argument of type 'NoneType' is not iterable.
        # Validate the lower layer first and translate that into an actionable
        # scanner error.
        try:
            import iio  # pylibiio bindings
            _ = getattr(iio, "version", None)
        except Exception as exc:
            msg = str(exc)
            extra = ""
            if isinstance(exc, TypeError) and "NoneType" in msg and "iterable" in msg:
                extra = (
                    "\n\nThis is the common Windows libiio/Python binding failure. "
                    "Install the native Analog Devices LibIIO Windows package, then in the SAME "
                    "Python environment run:\n"
                    "  python -m pip install -U pylibiio pyadi-iio\n"
                    "Then run diagnose_pluto.py before starting the GUI."
                )
            raise RuntimeError(
                "The libiio Python layer could not be loaded, so the scanner cannot talk to Pluto. "
                f"Original error: {type(exc).__name__}: {exc}{extra}"
            ) from exc

        try:
            import adi
        except Exception as exc:
            raise RuntimeError(
                "pyadi-iio could not be imported. In this scanner's Python environment run: "
                "python -m pip install -U pylibiio pyadi-iio. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            self.sdr = adi.Pluto(uri=self.uri)
            self.sdr.rx_enabled_channels = [0]
            self.sdr.gain_control_mode_chan0 = self.gain_mode
            if self.gain_mode == "manual":
                self.sdr.rx_hardwaregain_chan0 = float(self.manual_gain_db)
            self.sdr.rx_buffer_size = 65536
            # Helpful defaults for receive-only spectral work.
            try:
                self.sdr._rxadc.set_kernel_buffers_count(1)
            except Exception:
                pass
        except Exception as exc:
            self.sdr = None
            msg = str(exc)
            extra = ""
            if isinstance(exc, TypeError) and "NoneType" in msg and "iterable" in msg:
                extra = (
                    "\n\nOn Windows this usually means libiio is not visible to the Python "
                    "environment even though pyadi-iio is installed. Run diagnose_pluto.py. "
                    "If it reports a libiio failure, install native LibIIO and then reinstall "
                    "pylibiio + pyadi-iio in this same environment."
                )
            raise RuntimeError(
                f"Could not connect to ADALM-Pluto at {self.uri}. "
                f"Original error: {type(exc).__name__}: {exc}{extra}"
            ) from exc

    def close(self) -> None:
        if self.sdr is not None:
            try:
                self.sdr.rx_destroy_buffer()
            except Exception:
                pass
        self.sdr = None

    def capture(self, target: ScanTarget, wide_snapshot_samples: int = 131072) -> np.ndarray:
        if self.sdr is None:
            raise RuntimeError("Pluto is not connected")

        sr = int(target.sample_rate)
        bw = int(min(target.rf_bw, 20e6))
        n = wide_snapshot_samples if sr >= 10e6 else 65536
        n = int(2 ** round(math.log2(max(16384, n))))

        # Retuning can leave stale data in the streaming buffer; destroy/recreate around major changes.
        self.sdr.rx_destroy_buffer()
        self.sdr.sample_rate = sr
        self.sdr.rx_rf_bandwidth = bw
        self.sdr.rx_lo = int(target.center_hz)
        self.sdr.rx_buffer_size = n
        time.sleep(target.dwell_s)

        # First read is discarded after retune for settling/stale-buffer avoidance.
        try:
            _ = self.sdr.rx()
        except Exception:
            pass
        x = np.asarray(self.sdr.rx(), dtype=np.complex64)
        if x.ndim > 1:
            x = x[0]
        x = x - np.mean(x)
        # pyadi returns ADC-scale integers as complex samples. Normalize without changing SNR.
        scale = np.percentile(np.abs(x), 99.9)
        if scale > 0:
            x = x / scale
        return x


def _safe_db(x: np.ndarray | float, floor: float = 1e-20) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(x, floor))


def compute_psd(x: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = min(8192, len(x))
    if nperseg < 1024:
        raise ValueError("Not enough samples")
    f, pxx = signal.welch(
        x,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        return_onesided=False,
        scaling="density",
    )
    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(np.maximum(np.real(pxx), 1e-20))
    return f, pxx


def _merge_regions(mask: np.ndarray, gap_bins: int) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    regions: List[Tuple[int, int]] = []
    start = prev = int(idx[0])
    for cur_v in idx[1:]:
        cur = int(cur_v)
        if cur - prev > gap_bins + 1:
            regions.append((start, prev))
            start = cur
        prev = cur
    regions.append((start, prev))
    return regions


def _time_features(x: np.ndarray) -> Tuple[float, float, float]:
    # Envelope statistics are calculated in short chunks to capture packet/burst structure.
    n = len(x)
    block = max(64, min(2048, n // 128))
    usable = (n // block) * block
    if usable < block * 4:
        return 0.0, 0.0, 0.0
    env = np.mean(np.abs(x[:usable]).reshape(-1, block), axis=1)
    med = float(np.median(env)) + 1e-12
    mad = float(np.median(np.abs(env - med))) + 1e-12
    threshold = med + 4.0 * 1.4826 * mad
    duty = float(np.mean(env > threshold))
    cv = float(np.std(env) / (np.mean(env) + 1e-12))
    q10, q90 = np.percentile(env, [10, 90])
    burst = float((q90 - q10) / (q90 + q10 + 1e-12))
    return float(np.clip(burst, 0, 3)), float(np.clip(cv, 0, 5)), float(np.clip(duty, 0, 1))


def detect_components(
    x: np.ndarray,
    target: ScanTarget,
    classifier: "RFClassifier",
    max_components: int = 8,
) -> Tuple[List[Detection], np.ndarray, np.ndarray]:
    f_off, pxx = compute_psd(x, target.sample_rate)
    psd_db = _safe_db(pxx)

    # Ignore outer 4% of each snapshot (analog/digital filter roll-off) and the central DC neighborhood.
    n = len(pxx)
    valid = np.ones(n, dtype=bool)
    edge = max(4, int(0.04 * n))
    valid[:edge] = False
    valid[-edge:] = False
    dc_hz = max(20e3, target.sample_rate * 0.0015)
    valid[np.abs(f_off) < dc_hz] = False

    noise_db = float(np.percentile(psd_db[valid], 35))
    threshold_db = noise_db + target.min_snr_db
    mask = valid & (psd_db > threshold_db)

    bin_hz = abs(f_off[1] - f_off[0])
    gap_bins = max(1, int(8e3 / max(bin_hz, 1)))
    regions = _merge_regions(mask, gap_bins)

    # Discard tiny isolated FFT spikes unless they are very strong.
    candidates: List[Tuple[float, int, int]] = []
    for a, b in regions:
        width_hz = (b - a + 1) * bin_hz
        peak_db = float(np.max(psd_db[a:b+1]))
        if width_hz < max(2 * bin_hz, 2e3) and peak_db < noise_db + 18:
            continue
        candidates.append((peak_db, a, b))
    candidates.sort(reverse=True)

    burst, env_cv, duty = _time_features(x)
    detections: List[Detection] = []
    ts = datetime.now().isoformat(timespec="seconds")

    for peak_db, a, b in candidates[:max_components]:
        seg_p = pxx[a:b+1]
        seg_db = psd_db[a:b+1]
        seg_f = f_off[a:b+1]
        if len(seg_p) == 0:
            continue

        weights = np.maximum(seg_p - np.percentile(pxx[valid], 35), 0)
        if np.sum(weights) > 0:
            offset_hz = float(np.sum(seg_f * weights) / np.sum(weights))
        else:
            offset_hz = float(seg_f[np.argmax(seg_p)])
        signal_freq_hz = target.center_hz + offset_hz

        # Width of the connected above-threshold region is conservative. Expand slightly to avoid undercounting.
        bandwidth_hz = float(max(bin_hz, (b - a + 1) * bin_hz))
        snr_db = float(peak_db - noise_db)
        norm_p = seg_p / (np.mean(seg_p) + 1e-20)
        flatness = float(np.exp(np.mean(np.log(np.maximum(seg_p, 1e-20)))) / (np.mean(seg_p) + 1e-20))
        peakiness_db = float(np.max(seg_db) - np.mean(seg_db))
        kurt = float(stats.kurtosis(norm_p, fisher=False, bias=False)) if len(norm_p) >= 8 else 3.0
        if not np.isfinite(kurt):
            kurt = 3.0
        kurt = float(np.clip(kurt, 0, 100))

        features = {
            "freq_mhz": signal_freq_hz / 1e6,
            "bandwidth_khz": bandwidth_hz / 1e3,
            "snr_db": snr_db,
            "spectral_flatness": float(np.clip(flatness, 0, 1.2)),
            "peakiness_db": float(np.clip(peakiness_db, 0, 80)),
            "burstiness": burst,
            "envelope_cv": env_cv,
            "duty_cycle": duty,
            "spectral_kurtosis": kurt,
        }
        (
            label, conf, model_source, cnn_label, cnn_conf, cnn_accepted,
            cnn_rejection_reason, feature_label, feature_conf
        ) = classifier.predict_hybrid(
            features, target.hint, x, target, offset_hz, bandwidth_hz
        )
        note = classifier.note_for_prediction(label, features, target)
        if cnn_label:
            if cnn_accepted:
                note += f" AI={cnn_label} ({cnn_conf:.0%}) accepted; feature model={feature_label} ({feature_conf:.0%})."
            else:
                note += (
                    f" AI={cnn_label} ({cnn_conf:.0%}) REJECTED: {cnn_rejection_reason}; "
                    f"feature model={feature_label} ({feature_conf:.0%})."
                )
        detections.append(Detection(
            timestamp=ts,
            scan_name=target.name,
            center_freq_hz=target.center_hz,
            signal_freq_hz=signal_freq_hz,
            bandwidth_hz=bandwidth_hz,
            snr_db=snr_db,
            label=label,
            friendly_label=FRIENDLY_LABELS.get(label, label),
            confidence=conf,
            spectral_flatness=features["spectral_flatness"],
            peakiness_db=features["peakiness_db"],
            burstiness=burst,
            envelope_cv=env_cv,
            duty_cycle=duty,
            spectral_kurtosis=kurt,
            model_source=model_source,
            note=note,
            cnn_label=cnn_label,
            cnn_confidence=cnn_conf,
            cnn_accepted=cnn_accepted,
            cnn_rejection_reason=cnn_rejection_reason,
            feature_label=feature_label,
            feature_confidence=feature_conf,
        ))
    return detections, f_off + target.center_hz, psd_db


class RFClassifier:
    """Random-forest classifier with conservative physics/range post-checks.

    The included default model is trained on synthetic feature priors, not proprietary RF recordings.
    This makes it useful as a first-pass classifier while remaining honest about uncertainty.
    Real captures labeled by the user can be used to retrain the model with train_classifier.py.
    """

    def __init__(self, model_path: Optional[str] = None, cnn_path: Optional[str] = None, cnn_min_conf: float = 0.55):
        self.model = None
        self.model_source = "rules"
        self.classes_: Sequence[str] = []
        self.cnn = None
        self.cnn_path = cnn_path
        self.cnn_error = ""
        self.cnn_min_conf = float(cnn_min_conf)
        if model_path:
            self.load(model_path)
        if cnn_path:
            self.load_cnn(cnn_path)

    def load_cnn(self, path: str) -> None:
        try:
            if str(path).lower().startswith("hf:lwm-spectro"):
                from pretrained_lwm import LWMSpectroClassifier
                self.cnn = LWMSpectroClassifier()
            else:
                from cnn_rf import SpectrogramCNNClassifier
                self.cnn = SpectrogramCNNClassifier(path)
            self.cnn_path = path
            self.cnn_error = ""
        except Exception as exc:
            # AI support is optional: preserve the feature classifier if dependencies
            # are unavailable or a model cannot be loaded.
            self.cnn = None
            self.cnn_error = str(exc)

    def load(self, path: str) -> None:
        import joblib
        obj = joblib.load(path)
        self.model = obj["model"] if isinstance(obj, dict) else obj
        self.classes_ = list(getattr(self.model, "classes_", []))
        self.model_source = obj.get("source", "trained-model") if isinstance(obj, dict) else "trained-model"

    @staticmethod
    def vector(features: Dict[str, float]) -> np.ndarray:
        return np.array([[float(features[k]) for k in FEATURE_NAMES]], dtype=float)

    def predict(self, features: Dict[str, float], hint: str = "") -> Tuple[str, float, str]:
        freq = features["freq_mhz"]
        bw = features["bandwidth_khz"]
        snr = features["snr_db"]

        label = "unknown"
        conf = 0.0
        source = self.model_source
        if self.model is not None:
            proba = self.model.predict_proba(self.vector(features))[0]
            idx = int(np.argmax(proba))
            label = str(self.model.classes_[idx])
            conf = float(proba[idx])

        # Conservative physical/range guards. These reduce attractive-but-wrong ML labels.
        if 87.5 <= freq <= 108.5 and 40 <= bw <= 350:
            if label != "broadcast_fm" or conf < 0.70:
                label, conf, source = "broadcast_fm", max(conf, 0.72), source + "+band-guard"
        elif 2395 <= freq <= 2490:
            if bw >= 7000:
                label, conf, source = "wifi_24", max(conf, 0.80), source + "+bandwidth-guard"
            elif 300 <= bw <= 3500:
                # BLE vs Zigbee cannot always be resolved from a single short PSD snapshot.
                # Use proximity to 802.15.4 channel centers as a weak clue; keep confidence capped.
                zigbee_centers = np.array([2405 + 5 * k for k in range(16)], dtype=float)
                near_zigbee = float(np.min(np.abs(zigbee_centers - freq))) <= 1.2
                if near_zigbee and features["spectral_flatness"] > 0.20:
                    label, conf, source = "zigbee_24", min(max(conf, 0.58), 0.74), source + "+2.4-guard"
                else:
                    label, conf, source = "bluetooth_fhss", min(max(conf, 0.55), 0.72), source + "+2.4-guard"
        elif 5150 <= freq <= 5900 and bw >= 7000:
            label, conf, source = "wifi_5", max(conf, 0.78), source + "+bandwidth-guard"
        elif 431 <= freq <= 436 and bw <= 1000:
            label, conf, source = "ism_433", max(conf, 0.65), source + "+band-guard"
        elif 901 <= freq <= 929 and bw <= 4000:
            label, conf, source = "ism_915", max(conf, 0.60), source + "+band-guard"
        elif (312 <= freq <= 318 or 387 <= freq <= 393) and bw <= 1000:
            label, conf, source = "garage_remote", max(conf, 0.55), source + "+band-guard"
        elif bw < 100:
            if conf < 0.65:
                label, conf, source = "narrowband", max(conf, 0.52), source + "+generic"
        elif bw > 5000:
            if conf < 0.65:
                label, conf, source = "wideband_digital", max(conf, 0.52), source + "+generic"

        # Confidence gate: weak detections should not receive authoritative labels.
        if snr < 10 and conf > 0.70:
            conf = 0.70
        if conf < 0.45:
            label = "unknown"
        return label, float(np.clip(conf, 0, 0.99)), source

    @staticmethod
    def _wideband_snapshot_metrics(iq: np.ndarray, sample_rate: float) -> Dict[str, float]:
        """Estimate *instantaneous* spectral width from short-time FFT frames.

        This is intentionally different from the overall span visited during a capture.
        Bluetooth/FHSS can hop across many MHz over time while remaining ~1 MHz wide at
        each instant. Wi-Fi OFDM, by contrast, occupies many MHz simultaneously during
        an active packet. The OOD gate uses the 90th percentile of per-frame contiguous
        occupied bandwidth to separate those cases conservatively.
        """
        x = np.asarray(iq, dtype=np.complex64).ravel()
        sr = float(sample_rate)
        if x.size < 2048 or sr <= 0:
            return {
                "instantaneous_bw_mhz_p90": 0.0,
                "instantaneous_bw_mhz_max": 0.0,
                "wide_frame_fraction": 0.0,
                "active_bin_fraction_p90": 0.0,
            }

        nperseg = 1024 if x.size >= 8192 else 512
        hop = nperseg // 2
        win = np.hanning(nperseg).astype(np.float32)
        bin_hz = sr / nperseg
        contiguous_widths = []
        active_fracs = []

        # Avoid pathological runtime on unusually huge buffers while sampling the
        # complete capture uniformly enough to see bursty Wi-Fi packets.
        starts = np.arange(0, x.size - nperseg + 1, hop, dtype=int)
        if starts.size > 256:
            picks = np.linspace(0, starts.size - 1, 256).astype(int)
            starts = starts[picks]

        for st in starts:
            frame = x[st:st+nperseg]
            spec = np.fft.fftshift(np.fft.fft(frame * win))
            p_lin = np.abs(spec) ** 2

            # Smooth over ~200 kHz before thresholding. Raw OFDM FFT bins are
            # intentionally noise-like, so bin-by-bin thresholding fragments a
            # genuinely continuous 16-20 MHz channel into tiny islands. Smoothing
            # reveals the occupied spectral envelope while keeping 1-2 MHz
            # Bluetooth/Zigbee-like emissions narrow.
            smooth_bins = max(3, int(round(200e3 / max(bin_hz, 1.0))))
            if smooth_bins % 2 == 0:
                smooth_bins += 1
            kernel = np.ones(smooth_bins, dtype=float) / smooth_bins
            p_smooth = np.convolve(p_lin, kernel, mode="same")
            p_db = 10.0 * np.log10(p_smooth + 1e-20)

            # Low percentile gives a stable noise/background estimate even when a
            # broad OFDM signal occupies most (but not all) of the 20 MHz capture.
            floor = float(np.percentile(p_db, 12.0))
            mask = p_db > (floor + 4.5)

            # Suppress edge bins and the Pluto DC neighborhood.
            edge = max(2, int(0.02 * nperseg))
            mask[:edge] = False
            mask[-edge:] = False
            dc = nperseg // 2
            dc_guard = max(1, int(45e3 / max(bin_hz, 1.0)))
            mask[max(0, dc-dc_guard):min(nperseg, dc+dc_guard+1)] = False

            # Bridge very small OFDM notches but not MHz-scale gaps between FHSS hops.
            gap_bins = max(1, int(120e3 / max(bin_hz, 1.0)))
            if np.any(mask) and gap_bins > 1:
                idx = np.flatnonzero(mask)
                for a, b in zip(idx[:-1], idx[1:]):
                    if 1 < (b-a) <= gap_bins:
                        mask[a:b+1] = True

            active_fracs.append(float(np.mean(mask)))
            if not np.any(mask):
                contiguous_widths.append(0.0)
                continue

            # Largest contiguous occupied region in this time frame.
            padded = np.r_[False, mask, False]
            transitions = np.diff(padded.astype(np.int8))
            starts_r = np.flatnonzero(transitions == 1)
            ends_r = np.flatnonzero(transitions == -1)
            widths = ends_r - starts_r
            contiguous_widths.append(float(np.max(widths)) * bin_hz)

        if not contiguous_widths:
            return {
                "instantaneous_bw_mhz_p90": 0.0,
                "instantaneous_bw_mhz_max": 0.0,
                "wide_frame_fraction": 0.0,
                "active_bin_fraction_p90": 0.0,
            }

        widths = np.asarray(contiguous_widths, dtype=float)
        fracs = np.asarray(active_fracs, dtype=float)
        return {
            "instantaneous_bw_mhz_p90": float(np.percentile(widths, 90.0) / 1e6),
            "instantaneous_bw_mhz_max": float(np.max(widths) / 1e6),
            "wide_frame_fraction": float(np.mean(widths >= 6.0e6)),
            "active_bin_fraction_p90": float(np.percentile(fracs, 90.0)),
        }

    def predict_hybrid(
        self,
        features: Dict[str, float],
        hint: str,
        iq: np.ndarray,
        target: ScanTarget,
        component_offset_hz: float,
        component_bandwidth_hz: float,
    ) -> Tuple[str, float, str, str, float, bool, str, str, float]:
        """Fuse RF features with spectrogram AI, with an OOD/rejection gate.

        LWM-Spectro is a *closed-set* LTE/WiFi/5G classifier. It is therefore not
        allowed to vote on narrowband/FHSS/802.15.4-looking captures merely because
        its best of three choices is ``WiFi``. On dedicated Wi-Fi snapshots we first
        verify that the IQ contains genuinely wide *simultaneous* occupancy.
        """
        feature_label, feature_conf, feature_source = self.predict(features, hint)
        if self.cnn is None:
            return (
                feature_label, feature_conf, feature_source,
                "", 0.0, True, "", feature_label, feature_conf,
            )

        full_snapshot_ai = False
        ood_metrics: Dict[str, float] = {}
        try:
            from cnn_rf import map_cnn_to_rf_label
            is_lwm = getattr(self.cnn, "source", "").startswith("pretrained:LWM-Spectro")
            if is_lwm:
                full_snapshot_ai = (target.hint in {"wifi_24", "wifi_5"} and target.sample_rate >= 10e6)
                cp = self.cnn.predict(
                    iq,
                    target.sample_rate,
                    component_offset_hz,
                    component_bandwidth_hz,
                    force_wideband=full_snapshot_ai,
                )
            else:
                cp = self.cnn.predict(
                    iq,
                    target.sample_rate,
                    component_offset_hz,
                    component_bandwidth_hz,
                )
            cnn_raw = str(cp.label)
            cnn_conf = float(cp.confidence)

            map_freq_mhz = target.center_hz / 1e6 if full_snapshot_ai else features["freq_mhz"]
            map_bw_khz = target.rf_bw / 1e3 if full_snapshot_ai else features["bandwidth_khz"]
            mapped = map_cnn_to_rf_label(cnn_raw, map_freq_mhz, map_bw_khz)
        except Exception as exc:
            self.cnn_error = str(exc)
            return (
                feature_label, feature_conf, feature_source,
                "cnn-error", 0.0, False, str(exc), feature_label, feature_conf,
            )

        # ---------------- OOD / applicability rejection gate ----------------
        cnn_accepted = True
        rejection_reason = ""
        low_raw = cnn_raw.strip().lower()
        freq = float(features["freq_mhz"])
        component_bw_mhz = float(features["bandwidth_khz"]) / 1000.0

        if is_lwm and low_raw in {"wifi", "lte", "5g"}:
            # The household scanner only trusts LWM Wi-Fi evidence in Wi-Fi bands.
            # LTE/5G are kept as generic wideband-digital evidence elsewhere.
            if low_raw == "wifi" and not (2395 <= map_freq_mhz <= 2490 or 5150 <= map_freq_mhz <= 5900):
                cnn_accepted = False
                rejection_reason = "Wi-Fi output is outside a supported Wi-Fi band"

            if cnn_accepted and full_snapshot_ai:
                ood_metrics = self._wideband_snapshot_metrics(iq, target.sample_rate)
                p90_bw = ood_metrics["instantaneous_bw_mhz_p90"]
                max_bw = ood_metrics["instantaneous_bw_mhz_max"]
                wide_frac = ood_metrics["wide_frame_fraction"]
                active_frac = ood_metrics["active_bin_fraction_p90"]

                # Conservative Wi-Fi gate. A valid packet should show several MHz
                # occupied *at once*. Bluetooth hopping may span the full band over
                # time, but its per-frame contiguous width remains narrow. Zigbee is
                # ~2 MHz and likewise fails this gate.
                looks_wideband = (
                    p90_bw >= 6.0
                    and max_bw >= 7.0
                    and (wide_frac >= 0.02 or active_frac >= 0.22)
                )
                if low_raw == "wifi" and not looks_wideband:
                    cnn_accepted = False
                    rejection_reason = (
                        f"instantaneous occupied BW is only {p90_bw:.1f} MHz p90 "
                        f"({max_bw:.1f} MHz max); below the Wi-Fi wideband gate"
                    )

            elif cnn_accepted and low_raw == "wifi" and component_bw_mhz < 3.0:
                cnn_accepted = False
                rejection_reason = (
                    f"detected component is only {component_bw_mhz:.2f} MHz wide; "
                    "too narrow for a trusted Wi-Fi decision"
                )

        # If rejected, expose the raw AI evidence for transparency but give it
        # *zero influence* over the final household classification.
        if not cnn_accepted:
            source = feature_source + "+pretrained-ai-rejected-ood"
            return (
                feature_label, feature_conf, source,
                cnn_raw, cnn_conf, False, rejection_reason,
                feature_label, feature_conf,
            )

        # Synthetic starter data should never be treated as field-validated AI.
        effective_cnn = cnn_conf
        if "synthetic" in getattr(self.cnn, "source", "").lower():
            effective_cnn = min(effective_cnn, 0.74)

        final_label, final_conf = feature_label, feature_conf
        source = feature_source + "+" + getattr(self.cnn, "source", "spectrogram-cnn")

        if mapped == feature_label and mapped != "unknown" and effective_cnn >= self.cnn_min_conf:
            final_label = mapped
            final_conf = min(0.97, max(feature_conf, 0.50 * feature_conf + 0.50 * effective_cnn + 0.08))
            source += "+agreement"
        elif (
            full_snapshot_ai
            and mapped in {"wifi_24", "wifi_5"}
            and effective_cnn >= 0.80
        ):
            # The new OOD gate has already verified simultaneous wideband
            # occupancy. At this point a narrow connected PSD sub-component is
            # no longer a valid reason to prefer Bluetooth/Zigbee over a strong
            # Wi-Fi model result.
            final_label = mapped
            final_conf = min(0.93, max(0.80, effective_cnn))
            source += "+wideband-gated-wifi-override"
        elif mapped != "unknown" and effective_cnn >= 0.82 and feature_conf < 0.68:
            final_label = mapped
            final_conf = min(0.90, effective_cnn)
            source += "+cnn-override"
        elif mapped != "unknown" and mapped != feature_label and effective_cnn >= 0.68 and feature_conf >= 0.68:
            final_label = feature_label
            final_conf = min(feature_conf, 0.69)
            source += "+disagreement"
        elif feature_label == "unknown" and mapped != "unknown" and effective_cnn >= 0.72:
            final_label = mapped
            final_conf = min(0.78, effective_cnn)
            source += "+cnn-rescue"

        # Final plausibility checks.
        bw = features["bandwidth_khz"]
        if final_label in {"wifi_24", "wifi_5"} and bw < 3000 and not full_snapshot_ai:
            final_label, final_conf = ("wideband_digital" if bw >= 1000 else "unknown"), min(final_conf, 0.55)
            source += "+final-bw-guard"
        if final_label == "broadcast_fm" and not (87.5 <= freq <= 108.5):
            final_label, final_conf = "unknown", min(final_conf, 0.45)
            source += "+final-band-guard"

        return (
            final_label, float(np.clip(final_conf, 0, 0.99)), source,
            cnn_raw, cnn_conf, True, "", feature_label, feature_conf,
        )

    def note_for_prediction(self, label: str, features: Dict[str, float], target: ScanTarget) -> str:
        if label in {"bluetooth_fhss", "zigbee_24"}:
            return "Single-snapshot BLE/Zigbee classification is approximate; repeated hopping/channel behavior improves confidence."
        if label == "wifi_5":
            return "5 GHz reception depends on the Pluto frequency-extension modification and antenna/front-end response."
        if label == "broadcast_fm":
            return "Classified from FM-band location plus occupied-bandwidth/spectral features; no audio is decoded."
        if label == "unknown":
            return "No class reached the confidence threshold."
        return "Spectrum-shape classification only; no payload/protocol content is decoded."


def synthetic_training_data(seed: int = 570, samples_per_class: int = 900) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: List[List[float]] = []
    labels: List[str] = []

    def add(label: str, n: int, freq_rng, bw_rng, snr_rng, flat_rng, peak_rng, burst_rng, cv_rng, duty_rng, kurt_rng):
        for _ in range(n):
            vals = [
                rng.uniform(*freq_rng), rng.uniform(*bw_rng), rng.uniform(*snr_rng),
                rng.uniform(*flat_rng), rng.uniform(*peak_rng), rng.uniform(*burst_rng),
                rng.uniform(*cv_rng), rng.uniform(*duty_rng), rng.uniform(*kurt_rng),
            ]
            rows.append(vals); labels.append(label)

    n = samples_per_class
    add("broadcast_fm", n, (87.8,108.2),(80,260),(8,45),(0.10,0.55),(4,22),(0.02,0.35),(0.03,0.40),(0.00,0.25),(2,12))
    add("wifi_24", n, (2400,2485),(7000,22000),(7,45),(0.35,0.95),(1,12),(0.10,0.95),(0.15,1.8),(0.02,0.70),(1.8,8))
    add("wifi_5", n, (5150,5900),(7000,22000),(7,45),(0.35,0.95),(1,12),(0.10,0.95),(0.15,1.8),(0.02,0.70),(1.8,8))
    add("bluetooth_fhss", n, (2400,2483),(400,2200),(7,40),(0.18,0.80),(2,18),(0.25,1.8),(0.25,2.5),(0.02,0.55),(2,20))
    add("zigbee_24", n, (2404,2481),(700,3000),(7,40),(0.25,0.85),(2,16),(0.10,1.2),(0.15,1.7),(0.02,0.65),(2,15))
    add("ism_433", n, (432.0,435.8),(5,700),(7,45),(0.03,0.70),(3,30),(0.20,2.0),(0.20,2.8),(0.01,0.50),(2,30))
    add("ism_915", n, (902.0,928.0),(5,3000),(7,45),(0.04,0.85),(2,28),(0.10,2.0),(0.15,2.8),(0.01,0.70),(2,30))
    add("garage_remote", n, (312,393),(5,500),(7,45),(0.02,0.65),(3,32),(0.35,2.3),(0.30,3.2),(0.005,0.40),(2,35))
    add("narrowband", n, (70,6000),(2,90),(7,50),(0.01,0.55),(4,35),(0.02,1.5),(0.05,2.3),(0.00,0.60),(2,35))
    add("wideband_digital", n, (70,6000),(3000,20000),(7,45),(0.25,0.98),(1,18),(0.05,1.3),(0.10,2.0),(0.01,0.85),(1.5,15))
    return np.asarray(rows, dtype=float), np.asarray(labels)


def train_default_model(path: str) -> str:
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    X, y = synthetic_training_data()
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=570,
    )
    model.fit(X, y)
    obj = {
        "model": model,
        "features": FEATURE_NAMES,
        "source": "synthetic-prior-random-forest",
        "created": datetime.now().isoformat(timespec="seconds"),
        "warning": "Default model is trained on synthetic feature priors. Retrain with labeled local captures for higher reliability.",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


class CsvLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def write(self, detections: Iterable[Detection]) -> None:
        rows = [d.as_row() for d in detections]
        if not rows:
            return
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if not self._header_written:
                writer.writeheader(); self._header_written = True
            writer.writerows(rows)


class HouseScanner:
    def __init__(self, receiver: PlutoReceiver, classifier: RFClassifier, logger: Optional[CsvLogger] = None):
        self.receiver = receiver
        self.classifier = classifier
        self.logger = logger
        self.stop_requested = False

    def stop(self) -> None:
        self.stop_requested = True

    def scan(
        self,
        plan: Sequence[ScanTarget],
        loops: int = 1,
        on_target: Optional[Callable[[int, int, ScanTarget], None]] = None,
        on_result: Optional[Callable[[ScanTarget, List[Detection], np.ndarray, np.ndarray], None]] = None,
    ) -> List[Detection]:
        all_dets: List[Detection] = []
        self.stop_requested = False
        total = max(1, len(plan) * max(1, loops))
        index = 0
        for _loop in range(max(1, loops)):
            for target in plan:
                if self.stop_requested:
                    return all_dets
                index += 1
                if on_target:
                    on_target(index, total, target)
                try:
                    x = self.receiver.capture(target)
                    dets, freqs, psd_db = detect_components(x, target, self.classifier)
                except Exception as exc:
                    raise RuntimeError(
                        f"Pluto capture failed while scanning {target.name} at "
                        f"{target.center_hz/1e6:.3f} MHz: {type(exc).__name__}: {exc}"
                    ) from exc
                all_dets.extend(dets)
                if self.logger:
                    self.logger.write(dets)
                if on_result:
                    on_result(target, dets, freqs, psd_db)
        return all_dets


def save_iq_npz(path: str, x: np.ndarray, target: ScanTarget, label: Optional[str] = None) -> None:
    np.savez_compressed(
        path,
        iq=x.astype(np.complex64),
        center_hz=np.float64(target.center_hz),
        sample_rate=np.float64(target.sample_rate),
        rf_bw=np.float64(target.rf_bw),
        label="" if label is None else label,
        scan_name=target.name,
    )
