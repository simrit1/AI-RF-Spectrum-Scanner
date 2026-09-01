from __future__ import annotations
import numpy as np
from types import SimpleNamespace
from rf_core import RFClassifier, ScanTarget

class FakeLWM:
    source = "pretrained:LWM-Spectro/wi-lab"
    def predict(self, *args, **kwargs):
        return SimpleNamespace(label="WiFi", confidence=0.93)

def tone_like(sr=20e6, n=131072, bw=1.0e6):
    rng = np.random.default_rng(7)
    # Narrow complex Gaussian signal filtered in frequency domain.
    X = np.zeros(n, np.complex64)
    bins = max(8, int(n*bw/sr))
    c=n//2
    z=(rng.normal(size=bins)+1j*rng.normal(size=bins)).astype(np.complex64)
    X[c-bins//2:c-bins//2+bins]=z
    x=np.fft.ifft(np.fft.ifftshift(X)).astype(np.complex64)
    x += 0.00005*(rng.normal(size=n)+1j*rng.normal(size=n)).astype(np.complex64)
    return x

def wifi_like(sr=20e6, n=131072, bw=16e6):
    rng=np.random.default_rng(8)
    # Bursty broad OFDM-like noise: 16 MHz simultaneously occupied.
    X=np.zeros(n, np.complex64)
    bins=int(n*bw/sr); c=n//2
    z=(rng.normal(size=bins)+1j*rng.normal(size=bins)).astype(np.complex64)
    X[c-bins//2:c-bins//2+bins]=z
    x=np.fft.ifft(np.fft.ifftshift(X)).astype(np.complex64)
    x += 0.00005*(rng.normal(size=n)+1j*rng.normal(size=n)).astype(np.complex64)
    return x


def hopping_like(sr=20e6, n=131072):
    rng=np.random.default_rng(9)
    out=np.zeros(n, dtype=np.complex64)
    block=4096
    t=np.arange(block)/sr
    centers=[-7e6,-3e6,1e6,5e6,7e6]
    for i,st in enumerate(range(0,n-block+1,block)):
        fc=centers[i % len(centers)]
        # ~1 MHz noise-like hop shifted to a different center each block.
        z=(rng.normal(size=block)+1j*rng.normal(size=block)).astype(np.complex64)
        # Smooth by cumulative averaging to keep each hop relatively narrow.
        z=np.convolve(z, np.ones(20)/20, mode="same").astype(np.complex64)
        out[st:st+block]=z*np.exp(2j*np.pi*fc*t).astype(np.complex64)
    out += 0.01*(rng.normal(size=n)+1j*rng.normal(size=n)).astype(np.complex64)
    return out

def run_case(name, iq, feature_bw_khz, feature_label):
    clf=RFClassifier(None, None)
    clf.cnn=FakeLWM()
    # Bypass RF model dependency; force the intended local feature classification.
    clf.predict=lambda f,h="": (feature_label, 0.72, "test-feature")
    target=ScanTarget("2.4 GHz Wi-Fi snapshot", 2437e6, 20e6, 20e6, hint="wifi_24")
    feat={
        "freq_mhz":2437.0, "bandwidth_khz":feature_bw_khz, "snr_db":20,
        "spectral_flatness":0.5, "peakiness_db":5, "burstiness":0.5,
        "envelope_cv":0.5, "duty_cycle":0.3, "spectral_kurtosis":3,
    }
    out=clf.predict_hybrid(feat, target.hint, iq, target, 0, feature_bw_khz*1e3)
    final_label, _, _, ai_label, ai_conf, accepted, reason, *_ = out
    print(f"{name}: final={final_label}, AI={ai_label} {ai_conf:.0%}, accepted={accepted}")
    if reason: print("  reason:", reason)
    return final_label, accepted

if __name__ == "__main__":
    narrow_label, narrow_ok = run_case("narrow 1 MHz", tone_like(), 1000, "bluetooth_fhss")
    zigbee_label, zigbee_ok = run_case("Zigbee-like 2 MHz", tone_like(bw=2.0e6), 2000, "zigbee_24")
    hopping_label, hopping_ok = run_case("FHSS hopping across band", hopping_like(), 1000, "bluetooth_fhss")
    wide_label, wide_ok = run_case("wide 16 MHz", wifi_like(), 1200, "bluetooth_fhss")
    assert not narrow_ok and narrow_label == "bluetooth_fhss"
    assert not zigbee_ok and zigbee_label == "zigbee_24"
    assert not hopping_ok and hopping_label == "bluetooth_fhss"
    assert wide_ok and wide_label == "wifi_24"
    print("[OK] OOD gate rejects narrow WiFi guesses and accepts genuinely wide snapshots.")
