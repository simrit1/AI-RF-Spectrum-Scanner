# ADALM-Pluto House RF AI Scanner

A receive-only spectrum scanner for an ADALM-Pluto SDR. It sweeps common household RF bands, detects occupied spectral components, extracts RF features, and classifies them with a Random Forest model plus conservative RF sanity checks.

**It does not decode packets, Wi-Fi frames, Bluetooth traffic, audio, messages, identities, MAC addresses, or payload content.** It only analyzes RF energy and signal shape.

## What it can label

- Broadcast FM (88–108 MHz)
- Wi-Fi-like 2.4 GHz wide OFDM activity
- Wi-Fi-like 5 GHz wide OFDM activity (optional; depends on your extended Pluto and antenna/front-end response)
- Bluetooth/FHSS-like narrow 2.4 GHz activity
- Zigbee / IEEE 802.15.4-like narrow 2.4 GHz activity
- 433 MHz ISM device-like signals
- 902–928 MHz ISM / Z-Wave / LoRa-like signals
- 315/390 MHz remote-control-like signals
- Generic narrowband / wideband digital / unknown signals

A single PSD snapshot cannot always distinguish Bluetooth from Zigbee, or identify a proprietary ISM protocol. Low-confidence cases are deliberately labeled conservatively.

## Hardware

- ADALM-Pluto with antenna on **RX**
- USB connection to PC
- Your frequency-extension modification is useful for FM and optional 5 GHz scanning, but RF performance outside the stock supported range is not guaranteed.
- Use an antenna appropriate for the band you are scanning. One broadband antenna is convenient, but multiple band-specific antennas usually perform better.

## Windows setup

1. Install the official ADALM-Pluto USB/libiio driver package if Windows does not already recognize the Pluto correctly.
2. Install Python 3.10+.
3. Open Command Prompt in this folder.
4. Create and activate a virtual environment:

```bat
py -m venv .venv
.venv\Scripts\activate
```

5. Install dependencies:

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Confirm your Pluto is reachable. The normal USB-network URI is:

```text
ip:192.168.2.1
```

7. Start the GUI:

```bat
python house_rf_scanner.py
```

or double-click `start_scanner.bat` after the environment is installed.

## GUI operation

- **house_quick**: faster first-pass scan of FM, 315/390 MHz, 433 MHz, 902–928 MHz, and 2.4 GHz.
- **house_deep**: denser 902–928 MHz and 2.4 GHz sweep.
- **Include 5 GHz**: scans selected 20 MHz Wi-Fi channel centers above the Pluto's stock tuning range. Use only if your modification supports it and your antenna/front-end has useful response there.
- **Loops**: repeated scanning is useful for bursty devices such as BLE, Zigbee sensors, remotes, and smart-home devices.

The live graph shows the latest PSD snapshot. The table shows:

- estimated signal center frequency
- occupied bandwidth
- estimated SNR
- AI classification
- model confidence
- scan band

CSV files are written to `logs/` automatically.

## Command-line mode

```bat
python house_rf_scanner.py --cli --profile house_quick --loops 3
```

Deep scan:

```bat
python house_rf_scanner.py --cli --profile house_deep --loops 5
```

Include selected 5 GHz Wi-Fi channels:

```bat
python house_rf_scanner.py --cli --profile house_deep --include-5g --loops 3
```

Manual receive gain instead of AGC:

```bat
python house_rf_scanner.py --cli --gain-mode manual --gain 50 --loops 3
```

Custom sweep example (receive-only spectral classification):

```bat
python house_rf_scanner.py --cli --custom --start-mhz 400 --stop-mhz 500 --step-mhz 3 --sample-rate-msps 4 --loops 2
```

## Why Wi-Fi uses short snapshots

The Pluto RF channel can be configured wide enough to examine a 20 MHz Wi-Fi-like channel, but the USB 2.0 host link is a practical bottleneck for continuous high-rate streaming. The scanner therefore takes short high-rate snapshots at selected Wi-Fi centers, while lower-frequency sweeps use lower sample rates.

## The AI model

`models/house_rf_model.joblib` is a Random Forest trained from synthetic RF-feature priors. Features include:

- frequency
- occupied bandwidth
- SNR
- spectral flatness
- spectral peakiness
- burstiness
- envelope coefficient of variation
- estimated duty cycle
- spectral kurtosis

The classifier also applies physical sanity checks. For example, a 15 MHz-wide signal around 2.437 GHz can plausibly be Wi-Fi-like; a 30 kHz signal there should not be called Wi-Fi just because it is in the Wi-Fi band.

### Important limitation

The included model is a **starter model**, not a laboratory-trained universal RF recognizer. Real RF environments differ by antennas, front-end response, interference, distance, and device implementation. The strongest improvement is to collect and label examples from devices you control in your own home, then retrain.

## Retraining on your own labeled features

Prepare one or more CSV files containing the feature columns listed in `rf_core.py` plus a `label` column, then run:

```bat
python train_classifier.py my_labeled_features.csv --out models\my_house_model.joblib
```

Run with your model:

```bat
python house_rf_scanner.py --model models\my_house_model.joblib
```

## Improving Bluetooth vs Zigbee recognition

A single short capture can be ambiguous. Better recognition requires observing behavior over time:

- Bluetooth/BLE/FHSS-like activity tends to appear as short narrow emissions at changing frequencies.
- Zigbee/802.15.4 tends to occupy a roughly 2 MHz channel around one of the standard 2.4 GHz channel centers.
- Wi-Fi normally occupies a much wider channel and looks noise-like/OFDM-like in the PSD.

Run multiple loops and compare repeated detections rather than trusting one snapshot.

## Files

- `house_rf_scanner.py` — GUI and CLI application
- `rf_core.py` — Pluto capture, PSD detection, feature extraction, classifier and scan plans
- `models/house_rf_model.joblib` — included starter ML model
- `train_classifier.py` — retrain from your labeled feature CSVs
- `build_default_model.py` — rebuild the bundled starter model
- `requirements.txt` — Python dependencies
- `start_scanner.bat` — Windows launcher
- `config.example.json` — example configuration reference
- `logs/` — output CSVs

## Safety / privacy scope

Use this tool for spectrum awareness and RF experimentation with devices and environments you are authorized to examine. It intentionally does not contain protocol decoders or content extraction.

---

# CNN / Spectrogram AI upgrade

This version also includes a **spectrogram convolutional neural network (CNN)**. The scanner can now run as a hybrid classifier:

1. Pluto captures complex IQ.
2. The normal detector finds occupied spectral components.
3. Hand-engineered RF features are calculated (bandwidth, SNR, flatness, burstiness, etc.).
4. A time-frequency spectrogram is cropped around each detected component.
5. The CNN classifies the signal shape.
6. The feature model + RF band sanity checks and the CNN are fused into the final label.

The GUI now shows both the **Hybrid classification** and the raw **CNN evidence / confidence**. CSV logs also contain `cnn_label`, `cnn_confidence`, `feature_label`, and `feature_confidence` so disagreements are visible instead of hidden.

## CNN classes

The image model deliberately uses morphology classes that make sense from baseband spectrograms:

- `broadcast_fm`
- `wifi_ofdm`
- `bluetooth_fhss`
- `zigbee_like`
- `narrowband`
- `bursty_ism`
- `chirp_css`
- `wideband_digital`
- `unknown_noise`

Band information is fused afterward. For example, a strong `wifi_ofdm` morphology around 2.437 GHz maps to Wi-Fi 2.4 GHz-like, while the same morphology in the 5 GHz Wi-Fi region maps to Wi-Fi 5 GHz-like. A narrow signal is not allowed to become Wi-Fi merely because it is inside 2.4 GHz.

## Install CNN support

The normal requirements still support feature-only scanning. For CNN mode install the additional PyTorch dependency:

```bat
pip install -r requirements-cnn.txt
```

Then run normally:

```bat
python house_rf_scanner.py
```

The GUI status line should say something like:

```text
CNN ON: synthetic-prior-spectrogram-cnn
```

If PyTorch is not installed or the checkpoint cannot be loaded, the scanner automatically falls back to the Random-Forest/feature classifier rather than failing the entire scan.

Disable the CNN explicitly with:

```bat
python house_rf_scanner.py --no-cnn
```

Use another CNN checkpoint with:

```bat
python house_rf_scanner.py --cnn-model models\my_house_cnn.pt
```

## Included starter CNN

`models/house_rf_cnn.pt` is bundled so the hybrid path works immediately. It is trained from **synthetically generated RF waveform priors** representing FM-like modulation, OFDM-like signals, FHSS/GFSK-like signals, Zigbee-like QPSK activity, narrowband signals, bursty OOK/FSK-like ISM activity, chirps, generic wideband digital activity, and noise.

That is useful for software bring-up, but **synthetic validation accuracy is not real-world accuracy**. Antenna response, Pluto front-end behavior, multipath, interference, overlapping transmitters and real protocol implementations are different. The application therefore caps how strongly the bundled synthetic CNN can influence the final result.

The best model is one trained on real captures from devices you control in your own house.

## Collect your own CNN training data

Use `collect_cnn_samples.py`. Example: collect examples while your own 2.4 GHz Wi-Fi access point is producing known traffic:

```bat
python collect_cnn_samples.py ^
  --label wifi_ofdm ^
  --freq-mhz 2437 ^
  --sample-rate-msps 20 ^
  --rf-bw-mhz 20 ^
  --component-bw-khz 18000 ^
  --count 100
```

Example for a known 433.92 MHz household remote/sensor that you control:

```bat
python collect_cnn_samples.py ^
  --label bursty_ism ^
  --freq-mhz 433.92 ^
  --sample-rate-msps 4 ^
  --rf-bw-mhz 3.8 ^
  --component-bw-khz 250 ^
  --count 100
```

Example for a Bluetooth device you control:

```bat
python collect_cnn_samples.py ^
  --label bluetooth_fhss ^
  --freq-mhz 2440 ^
  --sample-rate-msps 4 ^
  --rf-bw-mhz 3.8 ^
  --component-bw-khz 1200 ^
  --count 150
```

Collect examples under multiple conditions: device close/far, idle/active, different rooms, different channels, and background-only captures. `unknown_noise` examples are especially important because they teach the CNN what **not** to classify.

## Train the CNN on your captures

If your dataset is under `cnn_dataset/`:

```text
cnn_dataset/
  wifi_ofdm/
  bluetooth_fhss/
  zigbee_like/
  bursty_ism/
  unknown_noise/
  ...
```

train with:

```bat
python train_cnn.py cnn_dataset --out models\my_house_cnn.pt --epochs 20
```

Then run:

```bat
python house_rf_scanner.py --cnn-model models\my_house_cnn.pt
```

`train_cnn.py` automatically creates spectrograms from IQ, applies mild time/frequency/noise augmentation, performs an 80/20 split, reports validation accuracy by epoch, and saves the best checkpoint.

For a trustworthy accuracy estimate, keep an additional set of captures from a different time/location completely out of training and validation, and test on those separately.

## Rebuild the synthetic starter CNN

```bat
python build_starter_cnn.py --out models\house_rf_cnn.pt
```

This is mainly for development/testing of the pipeline. Prefer a locally trained real-capture model for actual identification.

## CNN-related files

- `cnn_rf.py` — spectrogram creation, CNN architecture, checkpoint loading and inference
- `models/house_rf_cnn.pt` — bundled synthetic-prior CNN
- `collect_cnn_samples.py` — save labeled receive-only IQ examples
- `train_cnn.py` — train a CNN on real labeled captures
- `build_starter_cnn.py` — regenerate the synthetic starter model
- `requirements-cnn.txt` — normal dependencies plus PyTorch

## What the CNN does not do

The CNN sees signal shape in IQ-derived spectrograms. It does not extract Wi-Fi SSIDs, MAC addresses, Bluetooth identities, packet contents, microphone audio, passwords, messages, or other payload data. The project remains a passive spectrum-awareness/classification tool.

---

## Recommended setup: PRETRAINED AI, no training required

This package now defaults to the public **wi-lab/LWM-Spectro** pretrained wireless spectrogram model from Hugging Face instead of requiring you to train the included starter CNN.

### What the pretrained model adds

LWM-Spectro is a released wireless foundation / Mixture-of-Experts model whose public checkpoint distinguishes **WiFi, LTE, and 5G** from 128x128 received-signal spectrograms. The house scanner uses it only on sufficiently wide signals, then combines its result with measured center frequency, occupied bandwidth, SNR, and household-band rules.

That means:

- 2.4 GHz + wide signal + pretrained model says WiFi -> **Wi-Fi 2.4 GHz-like**
- 5 GHz + wide signal + pretrained model says WiFi -> **Wi-Fi 5 GHz-like**
- LTE/5G evidence outside a plausible Wi-Fi band -> **wideband digital** rather than falsely claiming Wi-Fi
- FM broadcast remains identified by its broadcast-FM band and RF shape
- 433/915 MHz remotes/sensors remain identified by band, bandwidth and burst behavior
- narrow 2.4 GHz activity is conservatively reported as Bluetooth/Zigbee/FHSS-like rather than being forced through a WiFi/LTE/5G model

### Install once

```bat
python -m pip install --upgrade pip
pip install -r requirements-pretrained.txt
```

### Run

Double-click:

```text
start_scanner_pretrained.bat
```

or run:

```bat
python house_rf_scanner.py
```

The first time the pretrained classifier is actually needed, it automatically downloads the public `wi-lab/lwm-spectro` inference code and its `moe_checkpoint.pth` through `huggingface_hub`. The large demonstration tensors are explicitly excluded from the download. After the Hugging Face cache has the model, subsequent runs use the cached copy.

### No training command is required

You do **not** need to run `train_cnn.py`, `collect_cnn_samples.py`, or `train_classifier.py` for the recommended setup. Those files remain only for experimentation.

### Model coverage caveat

There is not one credible open pretrained model that labels every possible household RF protocol from arbitrary Pluto snapshots. LWM-Spectro is used where it has actual pretrained coverage (WiFi/LTE/5G); other household categories are handled conservatively with RF-domain evidence instead of pretending a model was trained for classes it has never seen.


## If Start Scan says `argument of type NoneType is not iterable`

This error commonly happens on Windows when Python can import `pyadi-iio` but cannot load the native **libiio** library. The scanner now detects this and writes a full traceback to `scanner_error.log`.

1. Run `run_diagnostic.bat`.
2. If the libiio/pylibiio test fails, install the native Analog Devices **LibIIO Windows** release.
3. In the same Python/venv used by the scanner, run `repair_python_iio_windows.bat` or:

```bat
python -m pip install -U pylibiio pyadi-iio
python diagnose_pluto.py
```

Do not install packages into a different Python environment than the one used to start `house_rf_scanner.py`.

## Fix: `Pretrained AI evidence = not_applicable` / `AI conf. = 0%`

Version AI-FIX changes the LWM applicability logic. The PSD detector can split a 20 MHz Wi-Fi OFDM channel into small connected regions; earlier versions then incorrectly skipped LWM because each region looked narrower than 3 MHz. On dedicated 2.4/5 GHz Wi-Fi snapshot targets, LWM now receives the complete 20 MHz IQ snapshot regardless of the per-component width estimate. Genuinely narrowband FM/433/915/BLE/Zigbee rows continue to show `N/A (narrowband)` because LWM-Spectro was not trained for those classes.

To verify the public model independently, run `test_pretrained_ai.bat`. A random spectrogram prediction is not meaningful as a radio classification; the purpose is only to confirm the checkpoint downloads, imports, and performs inference.

## LWM upstream compatibility fix

This build pins the public LWM-Spectro model artifacts and uses the authors' own
historical Hugging Face Space code revision where both
`mixture/train_embedding_router.py` and its imported
`mixture/train_top1_router.py` are present. It also downloads the three public
LTE/WiFi/5G expert checkpoints automatically. No training is required.

Run `test_pretrained_ai.bat` first. A successful setup prints:

    [OK] Public pretrained model loaded and inference ran.

The label printed by that particular sanity check is intentionally meaningless
because it uses random input; only successful model construction/inference is
being tested.

## OOD / false-Wi-Fi rejection update

This build adds an out-of-distribution (OOD) gate in front of the public LWM-Spectro WiFi/LTE/5G model.
LWM-Spectro is a closed-set classifier, so a Bluetooth or Zigbee signal can otherwise be forced into its
closest available class (often `WiFi`). The scanner now treats that raw AI output as *evidence*, not truth.

For dedicated 2.4/5 GHz Wi-Fi snapshots the scanner measures **instantaneous occupied bandwidth** across
short-time FFT frames. This is deliberately different from the total frequency span visited over time:

- Bluetooth/FHSS may hop through many MHz, but each hop remains narrow at a given instant.
- Zigbee / IEEE 802.15.4 is also narrow compared with a 20 MHz Wi-Fi channel.
- Active Wi-Fi OFDM occupies many MHz simultaneously.

A raw LWM `WiFi` result is rejected when the snapshot does not pass the wideband gate. In the GUI this is
shown as, for example:

    WiFi — REJECTED     92%*

The percentage with `*` is the model's raw closed-set confidence; it is **not used in the final label**.
The final classification then remains the household RF classifier result such as `Bluetooth/FHSS-like`,
`Zigbee / 802.15.4-like`, or `Unknown`. The CSV log also records `cnn_accepted` and
`cnn_rejection_reason` for auditability.

A local no-Pluto self-test is included:

    python test_ood_gate.py

Expected final line:

    [OK] OOD gate rejects narrow WiFi guesses and accepts genuinely wide snapshots.
