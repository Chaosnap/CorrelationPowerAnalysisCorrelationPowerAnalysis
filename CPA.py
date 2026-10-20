"""Coefficient-level CPA engine for experimental ML-KEM FPGA leakage analysis.

This is NOT an AES byte/S-box attack. For each trace, operands must be the
*known coefficient presented to the targeted multiplier* (typically u-hat in
the NTT domain), not unprocessed ciphertext bytes. The recovered candidate
is the secret operand at that multiplication site (often s-hat), not
necessarily an original small CBD secret coefficient.

The default hypothesis is HW((public_operand * secret_guess) mod 3329).
An optional multiplicative factor models a *verified* arithmetic conversion.
Use factor=1 unless RTL analysis justifies a different value. HD needs the
actual previous register state for each trace (optional NPZ 'previous').
"""

import numpy as np

Q = 3329
HW_MOD = np.array([bin(x).count("1") for x in range(Q)], dtype=np.uint8)
HW_BYTE = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


def _popcount(values):
    """Vectorized Hamming weight of non-negative 64-bit words."""
    v = np.asarray(values, dtype=np.uint64).copy()
    result = np.zeros(v.shape, dtype=np.uint8)
    for _ in range(8):
        result += HW_BYTE[(v & 255).astype(np.intp)]
        v >>= 8
    return result


def load_npz(path):
    """Load one-to-one aligned traces and known multiplier operands.

    Required: traces (n_traces, n_samples), operands (n_traces, n_coefficients).
    Optional: previous, with the same shape as operands, for HD analysis.
    """
    with np.load(path, allow_pickle=False) as data:
        required = ("traces", "operands")
        if not all(name in data.files for name in required):
            raise ValueError("NPZ must contain arrays named 'traces' and 'operands'.")
        traces = np.array(data["traces"], copy=True)
        operands = np.array(data["operands"], copy=True)
        previous = np.array(data["previous"], copy=True) if "previous" in data.files else None
    _validate(traces, operands, previous)
    return traces, operands, previous


def _validate(traces, operands, previous=None):
    if traces.ndim != 2 or operands.ndim != 2:
        raise ValueError("traces and operands must both be 2-D arrays.")
    if traces.shape[0] != operands.shape[0] or traces.shape[0] < 3:
        raise ValueError("At least three matching trace/operand rows are required.")
    if traces.shape[1] < 1 or operands.shape[1] < 1:
        raise ValueError("Input arrays must not have empty columns.")
    if not np.issubdtype(traces.dtype, np.number):
        raise ValueError("Traces must be numeric.")
    if not np.issubdtype(operands.dtype, np.integer):
        raise ValueError("Operands must be integer-valued NTT coefficients.")
    if not np.isfinite(traces).all():
        raise ValueError("Traces contain NaN or Infinity.")
    if previous is not None:
        if previous.shape != operands.shape or not np.issubdtype(previous.dtype, np.integer):
            raise ValueError("'previous' must be an integer array matching operands.")


def _normalized(x):
    """Column-centered data and per-column L2 norms (zero-safe)."""
    centered = x.astype(np.float64, copy=False) - np.mean(x, axis=0, dtype=np.float64)
    norm = np.linalg.norm(centered, axis=0)
    return centered, norm


class MLKEMCPA:
    """Vectorized, chunked Pearson CPA for one observed multiplier coefficient."""

    def __init__(self, factor=1, word_bits=16, key_chunk=128, sample_chunk=256):
        if word_bits not in (12, 16, 32):
            raise ValueError("word_bits must be 12, 16 or 32.")
        if key_chunk < 1 or sample_chunk < 1:
            raise ValueError("Chunk sizes must be positive.")
        self.factor = int(factor) % Q
        self.word_bits = int(word_bits)
        self.key_chunk = int(key_chunk)
        self.sample_chunk = int(sample_chunk)

    def _leakage(self, public_values, guesses, model, previous_values):
        product = (public_values[:, None] * guesses[None, :] * self.factor) % Q
        if model == "HW":
            return HW_MOD[product]
        if model == "HD":
            if previous_values is None:
                raise ValueError("HD requires the NPZ 'previous' register-state array.")
            mask = (1 << self.word_bits) - 1
            changed = np.bitwise_xor(product, previous_values[:, None] & mask)
            return _popcount(changed)
        raise ValueError("model must be 'HW' or 'HD'.")

    def analyze(self, traces, operands, coefficient=0, start=0, stop=None,
                model="HW", previous=None, top_n=10):
        """Return key ranking, peak sample, full POI profile, window stability.

        start inclusive, stop exclusive. All sample indexes in the result are
        in the original trace coordinate system.
        """
        traces = np.asarray(traces)
        operands = np.asarray(operands)
        previous = None if previous is None else np.asarray(previous)
        _validate(traces, operands, previous)

        coefficient = int(coefficient)
        start = int(start)
        stop = traces.shape[1] if stop is None else int(stop)
        if not 0 <= coefficient < operands.shape[1]:
            raise ValueError("Coefficient index outside operand array.")
        if not 0 <= start < stop <= traces.shape[1]:
            raise ValueError("Invalid sample window [start, stop).")
        if model not in ("HW", "HD"):
            raise ValueError("Model must be HW or HD.")
        if model == "HD" and previous is None:
            raise ValueError("HD requires NPZ array 'previous' (register state before write).")

        # int64 arithmetic prevents overflow of q-sized products and the factor.
        public = operands[:, coefficient].astype(np.int64) % Q
        prev = None if previous is None else previous[:, coefficient].astype(np.int64)
        sampled = np.asarray(traces[:, start:stop], dtype=np.float64)
        trace_c, trace_norm = _normalized(sampled)
        if not np.any(trace_norm > 0):
            raise ValueError("All selected sample columns are constant.")

        all_keys = np.arange(Q, dtype=np.int64)
        scores = np.zeros(Q, dtype=np.float64)
        peak_index = np.zeros(Q, dtype=np.int32)
        signed_peak = np.zeros(Q, dtype=np.float64)
        profile = np.zeros(sampled.shape[1], dtype=np.float64)
        valid_guesses = np.zeros(Q, dtype=bool)

        for ka in range(0, Q, self.key_chunk):
            kb = min(ka + self.key_chunk, Q)
            keys = all_keys[ka:kb]
            hypothesis = self._leakage(public, keys, model, prev)
            h_c, h_norm = _normalized(hypothesis)
            valid_guesses[ka:kb] = h_norm > 0

            for ta in range(0, sampled.shape[1], self.sample_chunk):
                tb = min(ta + self.sample_chunk, sampled.shape[1])
                numerator = h_c.T @ trace_c[:, ta:tb]
                denom = h_norm[:, None] * trace_norm[None, ta:tb]
                corr = np.divide(numerator, denom, out=np.zeros_like(numerator),
                                 where=denom > 0)
                np.clip(corr, -1.0, 1.0, out=corr)
                magnitudes = np.abs(corr)
                profile[ta:tb] = np.maximum(profile[ta:tb], magnitudes.max(axis=0))
                local_idx = magnitudes.argmax(axis=1)
                local_corr = corr[np.arange(kb - ka), local_idx]
                improve = np.abs(local_corr) > scores[ka:kb]
                scores[ka:kb][improve] = np.abs(local_corr[improve])
                signed_peak[ka:kb][improve] = local_corr[improve]
                peak_index[ka:kb][improve] = ta + local_idx[improve]

        if not np.any(scores > 0):
            raise ValueError("No non-constant leakage hypothesis correlated with these traces.")
        ranking = np.argsort(-scores, kind="stable")
        top_n = min(Q, max(1, int(top_n)))
        best = int(ranking[0])
        peak_sample = start + int(peak_index[best])

        # Re-score every key in the same four windows surrounding the global POI.
        # This mirrors the comparison strategy, not the exact implementation,
        # of anshupriyan's Source_code.py.
        stable_counts = np.zeros(Q, dtype=np.int32)
        window_top = {}
        for radius in (1, 2, 3, 4):
            a = max(0, peak_index[best] - radius)
            b = min(sampled.shape[1], peak_index[best] + radius + 1)
            local_t, local_t_norm = _normalized(sampled[:, a:b])
            window_scores = np.zeros(Q, dtype=np.float64)
            for ka in range(0, Q, self.key_chunk):
                kb = min(ka + self.key_chunk, Q)
                h = self._leakage(public, all_keys[ka:kb], model, prev)
                hc, hn = _normalized(h)
                denom = hn[:, None] * local_t_norm[None, :]
                c = np.divide(hc.T @ local_t, denom,
                              out=np.zeros((kb - ka, b - a), dtype=np.float64),
                              where=denom > 0)
                window_scores[ka:kb] = np.sum(c * c, axis=1)
            leaders = np.argsort(-window_scores, kind="stable")[:top_n]
            stable_counts[leaders] += 1
            window_top[radius] = leaders.tolist()

        return {
            "coefficient": coefficient,
            "model": model,
            "factor": self.factor,
            "n_traces": int(traces.shape[0]),
            "start": start,
            "stop": stop,
            "best_key": best,
            "peak_corr": float(signed_peak[best]),
            "peak_sample": peak_sample,
            "top_keys": [
                {"key": int(k), "abs_corr": float(scores[k]),
                 "signed_corr": float(signed_peak[k]),
                 "sample": int(start + peak_index[k]),
                 "stability": int(stable_counts[k])}
                for k in ranking[:top_n]
            ],
            "correlation_by_key": scores,
            "poi_profile": profile,
            "window_top_keys": window_top,
        }
