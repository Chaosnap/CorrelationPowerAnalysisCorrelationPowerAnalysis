"""Minimal desktop GUI for the ML-KEM PWM/BaseMul HD-CPA engine.

This GUI is a thin wrapper around Main.analyze_file; no attack maths here.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from Main import analyze_file, plot_result


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML-KEM-768 BaseMul HD-CPA")
        self.geometry("760x650")
        self.result = None
        self.events = queue.Queue()
        self.fields = {}
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(frame, text="ML-KEM-768 PWM / BaseMul — HD Pearson CPA",
                  font=("", 15, "bold")).grid(row=row, column=0, columnspan=3,
                                               sticky="w", pady=(0, 12))
        row += 1
        row = self._file_row(frame, row, "HDF5 dataset", "dataset", "*.h5 *.hdf5")
        row = self._file_row(frame, row, "Candidate pairs (optional)", "candidates",
                             "*.npy *.csv")
        row = self._field(frame, row, "Target component (r0 / r1 / mul)",
                          "component", "r0")
        row = self._field(frame, row, "Guess (s0 or s1; if no candidate file)",
                          "guess", "s0")
        row = self._field(frame, row, "Known other secret coefficient",
                          "fixed_other", "")
        row = self._field(frame, row, "Zeta (optional if in HDF5)",
                          "zeta", "")
        row = self._field(frame, row, "Register width (bits)", "word_bits", "16")
        row = self._field(frame, row, "Start sample (optional)", "start", "")
        row = self._field(frame, row, "Stop sample (optional)", "stop", "")
        row = self._field(frame, row, "Max traces (optional)", "max_traces", "")
        row = self._field(frame, row, "Ground truth pair (optional: s0,s1)",
                          "true_pair", "")

        ttk.Label(frame, text=("HD uses /previous_register from HDF5. The other "
                               "coefficient must be known for 1D scans. "
                               "Match register value and zeta to YOUR RTL."),
                  wraplength=700, foreground="#744d11").grid(
                      row=row, column=0, columnspan=3, sticky="w", pady=(8, 10))
        row += 1

        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=3, sticky="w")
        self.start_button = ttk.Button(buttons, text="Run HD-CPA",
                                       command=self._start)
        self.start_button.pack(side="left", padx=(0, 10))
        self.plot_button = ttk.Button(buttons, text="Show correlation plots",
                                      command=self._plot, state="disabled")
        self.plot_button.pack(side="left")
        row += 1

        self.status = tk.StringVar(value="Load a prepared HDF5 dataset to begin.")
        ttk.Label(frame, textvariable=self.status, wraplength=710).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 8))
        row += 1

        self.table = ttk.Treeview(frame, columns=("rank", "s0", "s1", "corr"),
                                  show="headings", height=9)
        for name, title, width in [
            ("rank", "Rank", 65), ("s0", "s0 guess", 110),
            ("s1", "s1 guess", 110), ("corr", "Peak |rho|", 150)
        ]:
            self.table.heading(name, text=title)
            self.table.column(name, width=width, anchor="center")
        self.table.grid(row=row, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(row, weight=1)

    def _file_row(self, parent, row, label, key, formats):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        variable = tk.StringVar()
        self.fields[key] = variable
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(parent, text="Browse",
                   command=lambda: variable.set(
                       filedialog.askopenfilename(
                           filetypes=[("Supported files", formats),
                                      ("All files", "*.*")]) or variable.get())
                   ).grid(row=row, column=2, pady=3)
        return row + 1

    def _field(self, parent, row, label, key, initial):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        variable = tk.StringVar(value=initial)
        self.fields[key] = variable
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=6, pady=3)
        return row + 1

    def _start(self):
        def val(name):
            return self.fields[name].get().strip()

        def optional_int(name):
            return int(val(name)) if val(name) else None

        try:
            true = None
            if val("true_pair"):
                pieces = [int(v.strip()) for v in val("true_pair").split(",")]
                if len(pieces) != 2:
                    raise ValueError("Ground truth must have two comma-separated integers")
                true = pieces
            config = dict(
                dataset=val("dataset"), candidates_path=val("candidates") or None,
                guess=val("guess"), fixed_other=optional_int("fixed_other"),
                component=val("component"), zeta_override=optional_int("zeta"),
                word_bits=int(val("word_bits")),
                start=optional_int("start"), stop=optional_int("stop"),
                max_traces=optional_int("max_traces"), true_pair=true)
            if not config["dataset"]:
                raise ValueError("Select an HDF5 dataset first")
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.start_button.config(state="disabled")
        self.plot_button.config(state="disabled")
        self.status.set("Running coefficient-pair HD CPA...")
        self.result = None

        def work():
            try:
                self.events.put(("success", analyze_file(**config)))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        threading.Thread(target=work, daemon=True).start()
        self.after(100, self._poll)

    def _poll(self):
        try:
            kind, value = self.events.get_nowait()
        except queue.Empty:
            self.after(100, self._poll)
            return
        self.start_button.config(state="normal")
        if kind == "error":
            self.status.set("Analysis failed; check dataset and register mapping.")
            messagebox.showerror("CPA error", value)
            return
        self.result = value
        self.plot_button.config(state="normal")
        self.status.set(
            f"{value['trace_count']} traces | best pair {value['best_pair']} | "
            f"peak |rho| {value['best_score']:.5f} at sample "
            f"{value['best_sample_absolute']} | "
            f"ground-truth rank: {value.get('true_pair_rank', 'not supplied')}")
        for item in self.table.get_children():
            self.table.delete(item)
        for i, idx in enumerate(value["ranking"][:10], 1):
            pair = value["candidate_pairs"][idx]
            self.table.insert("", "end", values=(
                i, int(pair[0]), int(pair[1]),
                f"{value['peak_correlation'][idx]:.6f}"))

    def _plot(self):
        if self.result is not None:
            plot_result(self.result)


if __name__ == "__main__":
    App().mainloop()
