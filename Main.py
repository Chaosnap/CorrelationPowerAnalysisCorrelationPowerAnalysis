import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CPA import CPA
from ToolBox import ToolBox


def load_matrix(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError("NPZ input must contain exactly one array.")
        return data[data.files[0]]

    return pd.read_csv(path, header=None).to_numpy()


def parse_coefficients(text):
    result = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return result


def plot_result(result):
    coeff = result["coeff"]

    fig, ax = plt.subplots()
    ax.plot(result["poi_curve"])
    ax.axvline(result["global_poi"], linestyle="--")
    ax.set_xlabel("Sample")
    ax.set_ylabel("max |Pearson r|")
    ax.set_title(f"Coefficient {coeff}: POI search")
    ax.grid()
    plt.show()

    fig, ax = plt.subplots()
    relations = result["max_corr_per_guess"]
    ax.plot(np.arange(len(relations)), np.abs(relations))
    ax.axvline(result["best_guess"], linestyle="--")
    ax.set_xlabel("Secret coefficient guess")
    ax.set_ylabel("max |Pearson r|")
    ax.set_title(
        f"Coefficient {coeff}: best={result['best_guess']} "
        f"(r={result['peak_corr']:.4f})"
    )
    ax.grid()
    plt.show()


def build_parser():
    p = argparse.ArgumentParser(
        description="Coefficient-level CPA for Kyber / ML-KEM PWM/BaseMul leakage."
    )
    p.add_argument("--traces", required=True, help="Trace matrix: CSV, NPY or single-array NPZ.")
    p.add_argument("--inputs", required=True, help="Public coefficient matrix: CSV, NPY or NPZ.")
    p.add_argument(
        "--coeff",
        default="0",
        help="Coefficient indices, e.g. 0 or 0,1,2 or 0-9.",
    )
    p.add_argument("--previous", help="Previous register values for HD model.")
    p.add_argument("--model", choices=["HW", "HD"], default="HW")
    p.add_argument("--q", type=int, default=3329)
    p.add_argument(
        "--factor",
        type=int,
        default=1,
        help="Optional multiplicative domain factor. Use 1 for the direct Barrett RTL baseline.",
    )
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--chunk", type=int, default=256, help="Number of key guesses per vectorized block.")
    p.add_argument("--num-traces", type=int, default=0, help="0 means use all traces.")
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument("--sample-end", type=int, default=0, help="0 means use all samples after start.")
    p.add_argument("--align", action="store_true", help="FFT-align traces before CPA.")
    p.add_argument("--plot", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    traces = np.asarray(load_matrix(args.traces), dtype=np.float64)
    inputs = load_matrix(args.inputs)

    if traces.ndim == 1:
        traces = traces[:, None]
    if inputs.ndim == 1:
        inputs = inputs[:, None]

    if args.num_traces > 0:
        traces = traces[:args.num_traces]
        inputs = inputs[:args.num_traces]

    sample_end = args.sample_end if args.sample_end > 0 else traces.shape[1]
    traces = traces[:, args.sample_start:sample_end]

    if args.align:
        traces = ToolBox.AlignTracesFFT(traces)

    coeff_indices = parse_coefficients(args.coeff)

    cpa = CPA(
        keySize=len(coeff_indices),
        q=args.q,
        leakage_model=args.model,
        montgomery_factor=args.factor,
        top_n=args.top_n,
        guess_chunk=args.chunk,
    )
    cpa.SetTracesPointsAndPT(traces, inputs)

    if args.model == "HD":
        if not args.previous:
            raise SystemExit("--previous is required when --model HD is selected.")
        previous = load_matrix(args.previous)
        if args.num_traces > 0:
            previous = previous[:args.num_traces]
        cpa.SetPreviousValues(previous)

    recovered = cpa.Analyse(coeff_indices=coeff_indices)

    print("\n========== ML-KEM CPA RESULTS ==========")
    for result in cpa.GetResults():
        print(
            f"Coeff {result['coeff']:3d}: "
            f"guess={result['best_guess']:4d}, "
            f"peak={result['peak_corr']:+.6f}, "
            f"sample={result['peak_sample']}, "
            f"global_POI={result['global_poi']}"
        )
        print("  Top candidates:")
        for item in result["top_guesses"]:
            print(
                f"    k={item['guess']:4d}  "
                f"r={item['corr']:+.6f}  "
                f"sample={item['sample']:5d}  "
                f"stability={item['stability']}/{len(result['windows'])}"
            )

        if args.plot:
            plot_result(result)

    print("\nRecovered coefficients:", recovered)


if __name__ == "__main__":
    main()
