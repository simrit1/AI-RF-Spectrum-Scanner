"""One-command sanity check for the public pretrained LWM-Spectro backend."""
import traceback

print("LWM-Spectro pretrained AI sanity check")
print("No training is performed.")
print("First run downloads the MoE + LTE/WiFi/5G experts and compatible author code.")

try:
    import numpy as np
    from pretrained_lwm import LWMSpectroClassifier

    model = LWMSpectroClassifier()
    model._load()

    # Test the model directly with a 128x128 tensor. This avoids SDR/hardware
    # dependencies and proves that the public pretrained backend constructs and
    # executes in the current Python environment.
    torch = model.torch
    spec = torch.rand(128, 128)
    out = model.predictor.predict(spec, return_probabilities=True, return_routing=True)

    print("[OK] Public pretrained model loaded and inference ran.")
    print("     label:", out.get("label", out.get("predicted_class")))
    print("     confidence:", out.get("confidence"))
    print("     routing:", out.get("routing"))
    print("     code revision:", model.SPACE_CODE_REVISION[:8])
    print("     model revision:", model.MODEL_REVISION[:8])
    print("NOTE: the random-input label is meaningless; this only verifies the backend.")
except Exception as exc:
    print(f"[FAIL] Pretrained backend could not run: {type(exc).__name__}: {exc}")
    print("\nFull traceback:")
    traceback.print_exc()
    raise SystemExit(2)
