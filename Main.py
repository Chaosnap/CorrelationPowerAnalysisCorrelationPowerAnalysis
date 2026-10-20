"""ML-KEM coefficient-level CPA GUI and headless CLI.

Run GUI: python Main.py
Run CLI: python Main.py --cli data.npz --coeff 0 --model HW
"""
import argparse
import json
import queue
import threading

import numpy as np

from CPA import MLKEMCPA, load_npz


def cli(args):
    traces, operands, previous = load_npz(args.cli)
    result = MLKEMCPA(factor=args.factor, word_bits=args.word_bits).analyze(
        traces, operands, coefficient=args.coeff, start=args.start,
        stop=args.stop, model=args.model, previous=previous, top_n=args.top)
    print("ML-KEM coefficient CPA / hardware-intermediate model")
    print("Traces: {}, coefficient: {}, model: {}, factor: {}".format(
        result["n_traces"], result["coefficient"], result["model"], result["factor"]))
    print("Best candidate: {} | signed peak r: {:.5f} | sample: {}".format(
        result["best_key"], result["peak_corr"], result["peak_sample"]))
    print("Rank  Key  |Pearson r|  Peak sample  Stability (/4)")
    for i, row in enumerate(result["top_keys"], 1):
        print("{:4} {:4} {:11.5f} {:12} {:13}".format(
            i, row["key"], row["abs_corr"], row["sample"], row["stability"]))


def gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    class CPAApp:
        def __init__(self, root):
            self.root = root
            self.root.title("ML-KEM-768 | Coefficient CPA")
            self.root.geometry("1100x780")
            self.result = None
            self.data = None
            self.mailbox = queue.Queue()
            self.path = tk.StringVar()
            self.coeff = tk.StringVar(value="0")
            self.start = tk.StringVar(value="0")
            self.stop = tk.StringVar(value="")
            self.model = tk.StringVar(value="HW")
            self.factor = tk.StringVar(value="1")
            self.status = tk.StringVar(value="Load an ML-KEM dataset (.npz) to begin.")

            controls = ttk.Frame(root, padding=10)
            controls.pack(fill="x")
            ttk.Label(controls, text="NPZ dataset").grid(row=0, column=0, sticky="w")
            ttk.Entry(controls, textvariable=self.path, width=72).grid(
                row=0, column=1, columnspan=5, sticky="ew", padx=6)
            ttk.Button(controls, text="Browse / Load", command=self.load).grid(row=0, column=6, padx=4)

            items = [
                ("Coeff index", self.coeff),
                ("Start sample", self.start),
                ("Stop (exclusive)", self.stop),
                ("Model", self.model),
                ("Arithmetic factor", self.factor),
            ]
            for col, (label, var) in enumerate(items):
                ttk.Label(controls, text=label).grid(row=1, column=col, sticky="w", pady=(12, 0))
                if label == "Model":
                    widget = ttk.Combobox(controls, textvariable=var, values=("HW", "HD"),
                                          width=12, state="readonly")
                else:
                    widget = ttk.Entry(controls, textvariable=var, width=15)
                widget.grid(row=2, column=col, sticky="ew", padx=(0, 7))
            self.run_button = ttk.Button(controls, text="Run CPA", command=self.run)
            self.run_button.grid(row=2, column=6, sticky="ew", padx=4)
            controls.columnconfigure(1, weight=1)

            ttk.Label(root, textvariable=self.status, padding=(12, 6)).pack(
                fill="x", anchor="w")
            fig = Figure(figsize=(10, 5.1), dpi=100)
            self.ax_poi, self.ax_rank = fig.subplots(1, 2)
            fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.13, wspace=0.28)
            self.canvas = FigureCanvasTkAgg(fig, master=root)
            self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10)
            self.ax_poi.set_title("POI: maximum |Pearson r|")
            self.ax_rank.set_title("Coefficient hypotheses")
            self.canvas.draw()

            columns = ("rank", "key", "corr", "sample", "stable")
            self.table = ttk.Treeview(root, columns=columns, show="headings", height=10)
            for name, heading, width in [
                ("rank", "Rank", 65), ("key", "Candidate (0..3328)", 180),
                ("corr", "|Pearson r|", 160), ("sample", "Peak sample", 140),
                ("stable", "Top-10 windows (/4)", 170),
            ]:
                self.table.heading(name, text=heading)
                self.table.column(name, width=width, anchor="center")
            self.table.pack(fill="x", padx=12, pady=(0, 10))
            self.root.after(100, self.poll)

        def load(self):
            name = filedialog.askopenfilename(
                title="Select ML-KEM CPA NPZ",
                filetypes=[("NumPy NPZ dataset", "*.npz"), ("All files", "*.*")])
            if not name:
                return
            try:
                self.data = load_npz(name)
            except Exception as exc:
                messagebox.showerror("Dataset invalid", str(exc))
                return
            self.path.set(name)
            traces, operands, previous = self.data
            self.status.set(
                "Loaded {} traces x {} samples; {} operand columns; HD {}".format(
                    *traces.shape, operands.shape[1],
                    "available" if previous is not None else "requires 'previous' array"))
            self.stop.set(str(traces.shape[1]))

        def run(self):
            if self.data is None:
                messagebox.showerror("Missing dataset", "Load .npz before running CPA.")
                return
            try:
                coefficient = int(self.coeff.get())
                start = int(self.start.get())
                stop = int(self.stop.get()) if self.stop.get().strip() else None
                factor = int(self.factor.get())
                model = self.model.get()
                engine = MLKEMCPA(factor=factor)
            except ValueError as exc:
                messagebox.showerror("Invalid configuration", str(exc))
                return
            self.run_button.config(state="disabled")
            self.status.set("Computing candidate coefficients and Pearson correlations...")
            traces, operands, previous = self.data

            def worker():
                try:
                    result = engine.analyze(
                        traces, operands, coefficient=coefficient, start=start,
                        stop=stop, model=model, previous=previous)
                    self.mailbox.put(("ok", result))
                except Exception as exc:
                    self.mailbox.put(("error", str(exc)))
            threading.Thread(target=worker, daemon=True).start()

        def poll(self):
            try:
                while True:
                    kind, value = self.mailbox.get_nowait()
                    self.run_button.config(state="normal")
                    if kind == "ok":
                        self.display(value)
                    else:
                        self.status.set("Analysis failed: " + value)
                        messagebox.showerror("CPA error", value)
            except queue.Empty:
                pass
            self.root.after(100, self.poll)

        def display(self, result):
            self.result = result
            self.status.set(
                "Best candidate = {}  |  signed r = {:.5f}  |  peak sample = {}  |  "
                "N = {} traces  |  factor = {}".format(
                    result["best_key"], result["peak_corr"],
                    result["peak_sample"], result["n_traces"], result["factor"]))
            self.ax_poi.clear()
            x = np.arange(result["start"], result["stop"])
            self.ax_poi.plot(x, result["poi_profile"], linewidth=1)
            self.ax_poi.axvline(result["peak_sample"], linestyle="--", color="C1")
            self.ax_poi.set(title="POI: max |r| over 3329 guesses",
                            xlabel="Sample index", ylabel="max |Pearson r|")
            self.ax_rank.clear()
            keys = np.arange(len(result["correlation_by_key"]))
            self.ax_rank.plot(keys, result["correlation_by_key"], linewidth=0.8)
            self.ax_rank.axvline(result["best_key"], linestyle="--", color="C1")
            self.ax_rank.set(title="All secret coefficient candidates",
                             xlabel="Candidate value (mod 3329)", ylabel="max |Pearson r|")
            self.canvas.draw_idle()
            for item in self.table.get_children():
                self.table.delete(item)
            for i, item in enumerate(result["top_keys"], 1):
                self.table.insert("", "end", values=(
                    i, item["key"], "{:.5f}".format(item["abs_corr"]),
                    item["sample"], "{}/4".format(item["stability"])))

    root = tk.Tk()
    CPAApp(root)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="ML-KEM coefficient CPA")
    parser.add_argument("--cli", metavar="DATASET.npz", help="Run without a GUI")
    parser.add_argument("--coeff", type=int, default=0, help="Observed coefficient index")
    parser.add_argument("--start", type=int, default=0, help="First sample (inclusive)")
    parser.add_argument("--stop", type=int, default=None, help="Last sample (exclusive)")
    parser.add_argument("--model", choices=("HW", "HD"), default="HW")
    parser.add_argument("--factor", type=int, default=1,
                        help="Verified arithmetic factor, default 1 (never guess)")
    parser.add_argument("--word-bits", type=int, choices=(12, 16, 32), default=16)
    parser.add_argument("--top", type=int, default=10, help="Number of ranked guesses")
    args = parser.parse_args()
    if args.cli:
        cli(args)
    else:
        gui()


if __name__ == "__main__":
    main()
