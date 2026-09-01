"""Collect labeled receive-only IQ examples from an ADALM-Pluto for CNN training.

Use only with devices/signals you are authorized to examine. This script saves
raw IQ and metadata; it does not decode payloads.
"""
from __future__ import annotations
import argparse
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from rf_core import PlutoReceiver, ScanTarget
from cnn_rf import CNN_CLASS_NAMES

p = argparse.ArgumentParser(description="Collect labeled Pluto IQ examples for the spectrogram CNN")
p.add_argument("--label", required=True, choices=CNN_CLASS_NAMES)
p.add_argument("--freq-mhz", required=True, type=float, help="Center frequency in MHz")
p.add_argument("--sample-rate-msps", type=float, default=4.0)
p.add_argument("--rf-bw-mhz", type=float, default=3.8)
p.add_argument("--component-offset-khz", type=float, default=0.0, help="Approximate signal offset from tuned center")
p.add_argument("--component-bw-khz", type=float, default=1000.0, help="Approximate occupied bandwidth")
p.add_argument("--count", type=int, default=50)
p.add_argument("--interval", type=float, default=0.15)
p.add_argument("--uri", default="ip:192.168.2.1")
p.add_argument("--gain-mode", choices=["slow_attack","fast_attack","manual"], default="slow_attack")
p.add_argument("--gain", type=float, default=50.0)
p.add_argument("--out", default="cnn_dataset")
a = p.parse_args()

out = Path(a.out) / a.label
out.mkdir(parents=True, exist_ok=True)
receiver = PlutoReceiver(a.uri, a.gain_mode, a.gain)
target = ScanTarget(
    name=f"CNN training {a.label}",
    center_hz=a.freq_mhz * 1e6,
    sample_rate=a.sample_rate_msps * 1e6,
    rf_bw=a.rf_bw_mhz * 1e6,
    dwell_s=0.05,
    min_snr_db=0,
    hint="",
)
try:
    receiver.connect()
    for i in range(max(1, a.count)):
        iq = receiver.capture(target)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = out / f"{a.label}_{stamp}_{i:04d}.npz"
        np.savez_compressed(
            path,
            iq=iq.astype(np.complex64),
            sample_rate=np.float64(target.sample_rate),
            center_hz=np.float64(target.center_hz),
            component_offset_hz=np.float64(a.component_offset_khz * 1e3),
            component_bandwidth_hz=np.float64(a.component_bw_khz * 1e3),
            label=a.label,
        )
        print(f"[{i+1}/{a.count}] {path}")
        time.sleep(max(0.0, a.interval))
finally:
    receiver.close()
