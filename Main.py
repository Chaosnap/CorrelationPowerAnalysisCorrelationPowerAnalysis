"""Command-line entry point for ML-KEM-768 PWM/BaseMul HD-CPA.

Use python GUI.py for the optional desktop GUI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from CPA import CPA, Q


def load_dataset(path, start=None, stop=None, max_traces=None):
    """Load real traces + corresponding ciphertext and NTT operands from HDF5.

    The NTT-domain operands must be PRECOMPUTED from each ciphertext using
    the actual hardware decode/decompress/NTT conventions, then verified.
    This loader intentionally does not pretend raw ciphertext bytes are u0/u1.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install h5py (pip install -r requirements.txt)") from exc

    with h5py.File(path, "r") as f:
        required = ("traces", "ciphertexts", "ntt_u0", "ntt_u1",
                    "previous_register")
        missing = [name for name in required if name not in f]
        if missing:
            raise ValueError("HDF5 missing required datasets: " + ", ".join(missing))
        trace_ds = f["traces"]
        if trace_ds.ndim != 2:
            raise ValueError("/traces must be an N x T matrix")
        n, total_samples = trace_ds.shape
        if max_traces is not None:
            if max_traces < 3:
                raise ValueError("--max-traces must be at least 3")
            n = min(n, max_traces)
        first = 0 if start is None else start
        last = total_samples if stop is None else stop
        if first < 0 or last > total_samples or last - first < 3:
            raise ValueError("Invalid sample window; need at least three samples")
        traces = np.asarray(trace_ds[:n, first:last], dtype=np.float64)
        ct = f["ciphertexts"]
        if ct.ndim != 2 or ct.shape[0] < n or ct.shape[1] != 1088:
            raise ValueError("/ciphertexts must contain N rows of 1088 bytes for ML-KEM-768")
        # Verify that ciphertext rows and power trace rows remain aligned.
        for label in ("ntt_u0", "ntt_u1", "previous_register"):
            if f[label].shape != (trace_ds.shape[0],):
                raise ValueError(f"/{label} must have exactly one value per trace")
        u0 = np.asarray(f["ntt_u0"][:n], dtype=np.int64)
        u1 = np.asarray(f["ntt_u1"][:n], dtype=np.int64)
        prev = np.asarray(f["previous_register"][:n], dtype=np.int64)
        synthetic = bool(f.attrs.get("synthetic", False))
        zeta = None
        if "zeta" in f:
            zeta = np.asarray(f["zeta"][:n], dtype=np.int64)
        elif "zeta" in f.attrs:
            zeta = int(f.attrs["zeta"])
    return traces, u0, u1, prev, zeta, first, synthetic


def load_candidates(path=None, guess=None, fixed_other=None):
    """Enumerate a conditional one-coefficient search or read explicit pairs.

    Conditional search requires the OTHER coefficient to be known; this is
    a validation experiment, not an independent full two-coefficient recovery.
    """
    if path is not None:
        p = Path(path)
        pairs = (np.load(p, allow_pickle=False)
                 if p.suffix.lower() == ".npy"
                 else np.loadtxt(p, delimiter=",", dtype=np.int64))
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if guess is None or fixed_other is None:
        raise ValueError("Supply --candidates OR --guess and --fixed-other")
    if not 0 <= fixed_other < Q:
        raise ValueError("fixed other coefficient must be in 0..3328")
    values = np.arange(Q, dtype=np.int64)
    fixed = np.full(Q, fixed_other, dtype=np.int64)
    return (np.column_stack((values, fixed)) if guess == "s0"
            else np.column_stack((fixed, values)))


def analyze_file(dataset, candidates_path=None, guess=None, fixed_other=None,
                 component="r0", start=None, stop=None, max_traces=None,
                 word_bits=16, zeta_override=None, true_pair=None):
    traces, u0, u1, prev, zeta, sample_offset, synthetic = load_dataset(
        dataset, start=start, stop=stop, max_traces=max_traces)
    if zeta_override is not None:
        zeta = zeta_override
    candidates = load_candidates(candidates_path, guess, fixed_other)
    model = CPA(word_bits=word_bits, component=component)
    result = model.attack(
        traces, u0, u1, prev, candidates, zeta=zeta, true_pair=true_pair)
    result["best_sample_absolute"] = result["best_sample"] + sample_offset
    result["trace_count"] = len(traces)
    result["synthetic"] = synthetic
    return result


def plot_result(result):
    import matplotlib.pyplot as plt
    scores = result["peak_correlation"]
    order = result["ranking"]
    best_curve = result["correlation_matrix"][int(order[0])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(len(best_curve)), best_curve)
    axes[0].axvline(result["best_sample"], linestyle="--", alpha=0.6)
    axes[0].set(xlabel="Sample in cropped window",
                ylabel="Pearson correlation",
                title="Best candidate correlation curve")
    axes[1].plot(np.arange(len(scores)), scores, linewidth=0.8)
    axes[1].scatter([int(order[0])], [scores[int(order[0])]], zorder=3)
    axes[1].set(xlabel="Candidate index", ylabel="Peak |correlation|",
                title="Secret-pair candidate ranking")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="HD Pearson CPA for one ML-KEM-768 PWM/BaseMul register.")
    parser.add_argument("--dataset", required=True,
                        help="HDF5 containing traces, ciphertexts, NTT operands and previous register state")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--candidates", help="M x 2 candidate pairs (.npy or CSV)")
    choice.add_argument("--guess", choices=["s0", "s1"],
                        help="Vary one secret coefficient; requires --fixed-other")
    parser.add_argument("--fixed-other", type=int)
    parser.add_argument("--component", choices=["r0", "r1", "mul"], default="r0")
    parser.add_argument("--zeta", type=int, help="BaseMul zeta if not in HDF5")
    parser.add_argument("--word-bits", type=int, default=16)
    parser.add_argument("--start", type=int, help="First sample (0-based inclusive)")
    parser.add_argument("--stop", type=int, help="Last sample (0-based exclusive)")
    parser.add_argument("--max-traces", type=int)
    parser.add_argument("--true-pair", type=int, nargs=2, metavar=("S0", "S1"),
                        help="Optional laboratory ground truth for Key Rank")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = analyze_file(
        dataset=args.dataset, candidates_path=args.candidates,
        guess=args.guess, fixed_other=args.fixed_other,
        component=args.component, start=args.start, stop=args.stop,
        max_traces=args.max_traces, word_bits=args.word_bits,
        zeta_override=args.zeta, true_pair=args.true_pair)
    if result["synthetic"]:
        print("SYNTHETIC DATASET — not real FPGA key recovery")
    print(f"HD-CPA on {result['trace_count']} traces, BaseMul {result['component']}")
    print(f"Best pair: {result['best_pair']}; peak |rho|: {result['best_score']:.5f}; "
          f"sample: {result['best_sample_absolute']}")
    for rank, index in enumerate(result["ranking"][:10], 1):
        print(f"  {rank:2d}. {tuple(map(int, result['candidate_pairs'][index]))}: "
              f"|rho|={result['peak_correlation'][index]:.5f}")
    if "true_pair_rank" in result:
        print("Ground-truth rank:", result["true_pair_rank"])
    print("NOTE: A maximum correlation is a candidate ranking, not proof of key recovery.")
    if args.plot:
        plot_result(result)


if __name__ == "__main__":
    main()
