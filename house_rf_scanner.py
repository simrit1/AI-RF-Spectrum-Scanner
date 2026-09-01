from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from rf_core import (
    CsvLogger, HouseScanner, PlutoReceiver, RFClassifier,
    build_scan_plan, build_custom_plan, train_default_model,
)

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "house_rf_model.joblib"
DEFAULT_CNN_MODEL = "hf:lwm-spectro"


def ensure_model(path: Path) -> Path:
    if not path.exists():
        print(f"Building default AI model: {path}")
        train_default_model(str(path))
    return path


def cli_main(args) -> int:
    model_path = ensure_model(Path(args.model))
    cnn_path = None if args.no_cnn else str(args.cnn_model)
    classifier = RFClassifier(str(model_path), cnn_path=cnn_path, cnn_min_conf=args.cnn_min_conf)
    receiver = PlutoReceiver(args.uri, args.gain_mode, args.gain)
    log_path = Path(args.log) if args.log else HERE / "logs" / f"scan_{datetime.now():%Y%m%d_%H%M%S}.csv"
    logger = CsvLogger(str(log_path))
    scanner = HouseScanner(receiver, classifier, logger)

    if args.custom:
        plan = build_custom_plan(args.start_mhz * 1e6, args.stop_mhz * 1e6, args.step_mhz * 1e6, args.sample_rate_msps * 1e6)
    else:
        plan = build_scan_plan(args.profile, args.include_5g)

    try:
        print(f"Connecting to Pluto at {args.uri} ...")
        receiver.connect()
        print(f"Connected. Scanning {len(plan)} tunings per loop. Log: {log_path}")
        if classifier.cnn is not None:
            print(f"Hybrid AI: feature model + pretrained/spectrogram AI ({classifier.cnn.source})")
        elif not args.no_cnn:
            print(f"AI model unavailable; using feature model only. Reason: {classifier.cnn_error}")
        else:
            print("AI model disabled; using feature model only.")
        print("Spectrum classification only — no packet/audio/message decoding.\n")

        def on_target(i, total, target):
            print(f"[{i:3d}/{total:3d}] {target.name:24s} {target.center_hz/1e6:9.3f} MHz", end="", flush=True)

        def on_result(target, dets, freqs, psd):
            if not dets:
                print("   no confident component")
                return
            print()
            for d in dets:
                print(
                    f"    {d.signal_freq_hz/1e6:10.4f} MHz  "
                    f"BW {d.bandwidth_hz/1e3:8.1f} kHz  SNR {d.snr_db:5.1f} dB  "
                    f"{d.friendly_label:34s}  {100*d.confidence:4.0f}%"
                    + (
                        (f"  AI:{d.cnn_label} {100*d.cnn_confidence:.0f}%" if d.cnn_accepted
                         else f"  AI:{d.cnn_label} {100*d.cnn_confidence:.0f}% [REJECTED: {d.cnn_rejection_reason}]")
                        if d.cnn_label else ""
                    )
                )

        scanner.scan(plan, loops=args.loops, on_target=on_target, on_result=on_result)
        print(f"\nFinished. Results saved to {log_path}")
        return 0
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        receiver.close()


def gui_main(args) -> int:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    model_path = ensure_model(Path(args.model))
    cnn_path = None if args.no_cnn else str(args.cnn_model)
    classifier = RFClassifier(str(model_path), cnn_path=cnn_path, cnn_min_conf=args.cnn_min_conf)

    root = tk.Tk()
    root.title("ADALM-Pluto House RF AI Scanner")
    root.geometry("1380x820")
    root.minsize(1100, 680)

    q = queue.Queue()
    worker = {"thread": None, "scanner": None, "receiver": None}

    top = ttk.Frame(root, padding=8); top.pack(fill="x")
    ttk.Label(top, text="Pluto URI").grid(row=0, column=0, sticky="w")
    uri_var = tk.StringVar(value=args.uri)
    ttk.Entry(top, textvariable=uri_var, width=22).grid(row=0, column=1, padx=(4, 12))
    ttk.Label(top, text="Profile").grid(row=0, column=2, sticky="w")
    profile_var = tk.StringVar(value=args.profile)
    ttk.Combobox(top, textvariable=profile_var, values=["house_quick", "house_deep"], width=15, state="readonly").grid(row=0, column=3, padx=(4, 12))
    include5_var = tk.BooleanVar(value=args.include_5g)
    ttk.Checkbutton(top, text="Include 5 GHz (extended Pluto)", variable=include5_var).grid(row=0, column=4, padx=(0, 12))
    loops_var = tk.IntVar(value=max(1, args.loops))
    ttk.Label(top, text="Loops").grid(row=0, column=5)
    ttk.Spinbox(top, from_=1, to=9999, textvariable=loops_var, width=6).grid(row=0, column=6, padx=(4, 12))

    cnn_status = (f"AI ON: {classifier.cnn.source}" if classifier.cnn is not None
                  else ("AI OFF" if args.no_cnn else f"AI unavailable: {classifier.cnn_error}"))
    status_var = tk.StringVar(value=f"Ready — hybrid receive-only spectrum classifier — {cnn_status}")
    progress_var = tk.DoubleVar(value=0)

    btn_frame = ttk.Frame(top); btn_frame.grid(row=0, column=7, sticky="e")
    start_btn = ttk.Button(btn_frame, text="Start scan")
    stop_btn = ttk.Button(btn_frame, text="Stop", state="disabled")
    clear_btn = ttk.Button(btn_frame, text="Clear")
    start_btn.pack(side="left", padx=2); stop_btn.pack(side="left", padx=2); clear_btn.pack(side="left", padx=2)
    top.columnconfigure(7, weight=1)

    ttk.Progressbar(root, variable=progress_var, maximum=100).pack(fill="x", padx=8)
    ttk.Label(root, textvariable=status_var, padding=(8,4)).pack(fill="x")

    body = ttk.Panedwindow(root, orient="horizontal"); body.pack(fill="both", expand=True, padx=8, pady=4)
    left = ttk.Frame(body); right = ttk.Frame(body)
    body.add(left, weight=5); body.add(right, weight=4)

    columns = ("time","freq","label","conf","cnn","cnnconf","bw","snr","scan")
    tree = ttk.Treeview(left, columns=columns, show="headings", height=26)
    heads = {"time":"Time","freq":"Frequency (MHz)","label":"Hybrid classification","conf":"Confidence",
             "cnn":"Pretrained AI evidence","cnnconf":"AI conf.","bw":"BW (kHz)","snr":"SNR (dB)","scan":"Band/sweep"}
    widths = {"time":80,"freq":118,"label":205,"conf":82,"cnn":190,"cnnconf":76,"bw":88,"snr":72,"scan":145}
    for c in columns:
        tree.heading(c, text=heads[c]); tree.column(c, width=widths[c], anchor="center" if c not in {"label","scan"} else "w")
    ys = ttk.Scrollbar(left, orient="vertical", command=tree.yview); tree.configure(yscrollcommand=ys.set)
    tree.pack(side="left", fill="both", expand=True); ys.pack(side="right", fill="y")

    fig = Figure(figsize=(6,5), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlabel("Frequency (MHz)"); ax.set_ylabel("PSD (dB/Hz)"); ax.grid(True, alpha=0.25)
    line, = ax.plot([], [])
    canvas = FigureCanvasTkAgg(fig, master=right); canvas.get_tk_widget().pack(fill="both", expand=True)
    info_var = tk.StringVar(value="The plot shows the latest Pluto snapshot.\nNo payloads are decoded or stored.")
    ttk.Label(right, textvariable=info_var, justify="left", padding=8).pack(fill="x")

    log_path = HERE / "logs" / f"scan_{datetime.now():%Y%m%d_%H%M%S}.csv"

    def add_detection(d):
        # Do not display N/A as if it were a failed 0%-confidence prediction.
        if d.cnn_label == "not_applicable":
            ai_evidence, ai_conf = "N/A (narrowband)", "—"
        elif d.cnn_label == "cnn-error":
            ai_evidence, ai_conf = "AI ERROR", "—"
        elif d.cnn_label and not d.cnn_accepted:
            ai_evidence = f"{d.cnn_label} — REJECTED"
            ai_conf = f"{100*d.cnn_confidence:.0f}%*"
        elif d.cnn_label:
            ai_evidence, ai_conf = d.cnn_label, f"{100*d.cnn_confidence:.0f}%"
        else:
            ai_evidence, ai_conf = "—", "—"
        tree.insert("", 0, values=(
            d.timestamp[-8:], f"{d.signal_freq_hz/1e6:.5f}", d.friendly_label,
            f"{100*d.confidence:.0f}%", ai_evidence, ai_conf,
            f"{d.bandwidth_hz/1e3:.1f}", f"{d.snr_db:.1f}", d.scan_name,
        ))

    def process_queue():
        try:
            while True:
                item = q.get_nowait()
                kind = item[0]
                if kind == "progress":
                    _, i, total, target = item
                    progress_var.set(100*i/max(1,total))
                    status_var.set(f"Scanning {target.name} at {target.center_hz/1e6:.3f} MHz ({i}/{total})")
                elif kind == "result":
                    _, target, dets, freqs, psd = item
                    for d in dets:
                        add_detection(d)
                        if d.cnn_label == "cnn-error" and classifier.cnn_error:
                            status_var.set(f"Pretrained AI error: {classifier.cnn_error}")
                        elif d.cnn_label and not d.cnn_accepted:
                            status_var.set(f"AI evidence rejected as out-of-distribution: {d.cnn_rejection_reason}")
                    if len(freqs):
                        line.set_data(freqs/1e6, psd)
                        ax.relim(); ax.autoscale_view(); canvas.draw_idle()
                        info_var.set(f"Latest: {target.name} — {target.center_hz/1e6:.3f} MHz\nDetected components: {len(dets)} | {cnn_status}\nLog: {log_path}")
                elif kind == "error":
                    _, msg = item
                    status_var.set("Error")
                    messagebox.showerror("Pluto scanner", msg)
                elif kind == "done":
                    status_var.set(f"Finished — CSV log: {log_path}")
                    start_btn.config(state="normal"); stop_btn.config(state="disabled")
                    progress_var.set(100)
        except queue.Empty:
            pass
        root.after(100, process_queue)

    def run_worker():
        receiver = PlutoReceiver(uri_var.get().strip(), args.gain_mode, args.gain)
        logger = CsvLogger(str(log_path))
        scanner = HouseScanner(receiver, classifier, logger)
        worker["receiver"] = receiver; worker["scanner"] = scanner
        try:
            receiver.connect()
            plan = build_scan_plan(profile_var.get(), include5_var.get())
            scanner.scan(
                plan,
                loops=max(1, int(loops_var.get())),
                on_target=lambda i,t,target: q.put(("progress",i,t,target)),
                on_result=lambda target,dets,freqs,psd: q.put(("result",target,dets,freqs,psd)),
            )
        except Exception as exc:
            error_log = HERE / "scanner_error.log"
            try:
                error_log.write_text(traceback.format_exc(), encoding="utf-8")
            except Exception:
                pass
            q.put(("error", f"{exc}\n\nFull traceback saved to:\n{error_log}"))
            return
        finally:
            receiver.close()
        q.put(("done",))

    def start_scan():
        if worker["thread"] and worker["thread"].is_alive(): return
        progress_var.set(0); start_btn.config(state="disabled"); stop_btn.config(state="normal")
        status_var.set("Connecting to Pluto...")
        t = threading.Thread(target=run_worker, daemon=True); worker["thread"] = t; t.start()

    def stop_scan():
        if worker["scanner"]: worker["scanner"].stop()
        status_var.set("Stopping after current capture...")

    def clear():
        for iid in tree.get_children(): tree.delete(iid)
        line.set_data([], []); ax.relim(); ax.autoscale_view(); canvas.draw_idle()

    def on_close():
        if worker["scanner"]: worker["scanner"].stop()
        root.destroy()

    start_btn.config(command=start_scan); stop_btn.config(command=stop_scan); clear_btn.config(command=clear)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, process_queue)
    root.mainloop()
    return 0


def parser():
    p = argparse.ArgumentParser(description="Receive-only ADALM-Pluto household RF spectrum scanner with ML classification")
    p.add_argument("--uri", default="ip:192.168.2.1")
    p.add_argument("--profile", choices=["house_quick","house_deep"], default="house_quick")
    p.add_argument("--include-5g", action="store_true", help="Scan selected 5 GHz Wi-Fi channels (requires extended tuning and suitable RF response)")
    p.add_argument("--loops", type=int, default=1)
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="Feature/Random-Forest model")
    p.add_argument("--cnn-model", default=DEFAULT_CNN_MODEL, help="AI model: hf:lwm-spectro (default pretrained) or local CNN checkpoint (.pt)")
    p.add_argument("--cnn-min-conf", type=float, default=0.55, help="Minimum AI confidence used for agreement fusion")
    p.add_argument("--no-cnn", action="store_true", help="Disable pretrained/spectrogram AI and use feature model only")
    p.add_argument("--gain-mode", choices=["slow_attack","fast_attack","manual"], default="slow_attack")
    p.add_argument("--gain", type=float, default=50.0, help="Manual RX gain in dB when --gain-mode manual")
    p.add_argument("--log", default="")
    p.add_argument("--cli", action="store_true", help="Run terminal mode instead of GUI")
    p.add_argument("--custom", action="store_true", help="CLI only: custom start/stop sweep")
    p.add_argument("--start-mhz", type=float, default=70.0)
    p.add_argument("--stop-mhz", type=float, default=6000.0)
    p.add_argument("--step-mhz", type=float, default=3.0)
    p.add_argument("--sample-rate-msps", type=float, default=4.0)
    return p

if __name__ == "__main__":
    a = parser().parse_args()
    raise SystemExit(cli_main(a) if a.cli else gui_main(a))
