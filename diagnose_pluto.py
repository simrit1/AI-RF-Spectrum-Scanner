from __future__ import annotations

import argparse
import platform
import sys
import traceback


def main() -> int:
    p = argparse.ArgumentParser(description="Check Python/libiio/pyadi-iio/ADALM-Pluto connectivity")
    p.add_argument("--uri", default="ip:192.168.2.1")
    args = p.parse_args()

    print("ADALM-Pluto scanner diagnostic")
    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    print("OS:", platform.platform())
    print("URI:", args.uri)
    print()

    try:
        import iio
        print("[OK] pylibiio imported")
        print("     iio.version =", getattr(iio, "version", "<not exposed>"))
    except Exception as exc:
        print("[FAIL] libiio / pylibiio layer")
        print(f"       {type(exc).__name__}: {exc}")
        if isinstance(exc, TypeError) and "NoneType" in str(exc) and "iterable" in str(exc):
            print("\nThis matches the common Windows native-libiio-not-found failure.")
        print("\nWindows repair:")
        print("  1. Install the native Analog Devices LibIIO Windows release.")
        print("  2. In THIS Python environment run:")
        print("       python -m pip install -U pylibiio pyadi-iio")
        print("  3. Close/reopen Command Prompt, then rerun this diagnostic.")
        print("\nFull traceback:")
        traceback.print_exc()
        return 2

    try:
        import adi
        print("[OK] pyadi-iio imported", getattr(adi, "__version__", ""))
    except Exception as exc:
        print(f"[FAIL] pyadi-iio import: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 3

    sdr = None
    try:
        sdr = adi.Pluto(uri=args.uri)
        print("[OK] Pluto context opened")
        print("     sample_rate:", sdr.sample_rate)
        print("     rx_lo:", sdr.rx_lo)
        print("     rx_rf_bandwidth:", sdr.rx_rf_bandwidth)
        sdr.rx_enabled_channels = [0]
        sdr.rx_buffer_size = 4096
        x = sdr.rx()
        try:
            n = len(x)
        except Exception:
            n = -1
        print(f"[OK] RX capture succeeded ({n} samples)")
        print("\nYour Python -> libiio -> Pluto path is working.")
        return 0
    except Exception as exc:
        print(f"[FAIL] Pluto connection/capture: {type(exc).__name__}: {exc}")
        print("\nChecks:")
        print("  - Pluto USB cable connected and powered")
        print("  - URI is normally ip:192.168.2.1")
        print("  - Try opening http://192.168.2.1 in a browser")
        print("  - On Windows, ensure the Analog Devices Pluto/libiio drivers are installed")
        print("\nFull traceback:")
        traceback.print_exc()
        return 4
    finally:
        if sdr is not None:
            try:
                sdr.rx_destroy_buffer()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
