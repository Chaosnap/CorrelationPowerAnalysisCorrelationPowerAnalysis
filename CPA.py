"""ML-KEM-768 coefficient-pair CPA engine (PWM / BaseMul).

An explicitly *register-aware* Hamming Distance hypothesis:
    HD(previous_register, candidate-dependent BaseMul register value)

This is the canonical modular BaseMul model, NOT an automatically verified
gate-level/RTL model. Users must verify reduction, Montgomery representation,
signedness, and the exact targeted register before drawing hardware conclusions.

The module is intentionally GUI-independent.
"""
from __future__ import annotations

import numpy as np

Q = 3329
_BYTE_HW = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def _vector(values, n, name):
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = np.full(n, arr)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be scalar or shape ({n},)")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must contain integer register/coefficients")
    return arr.astype(np.int64)


def hamming_distance(previous, current, word_bits=16):
    """Count switching bits, masking BOTH states to the register width."""
    if not 1 <= word_bits <= 32:
        raise ValueError("word_bits must be in [1, 32]")
    mask = (1 << word_bits) - 1
    x = (np.asarray(previous, dtype=np.uint64)
         ^ np.asarray(current, dtype=np.uint64)) & np.uint64(mask)
    hw = np.zeros(x.shape, dtype=np.float64)
    for shift in range(0, word_bits, 8):
        hw += _BYTE_HW[((x >> np.uint64(shift)) & 255).astype(np.uint8)]
    return hw


def basemul_register_values(u0, u1, candidates, zeta=None, component="r0"):
    """Predict one canonical BaseMul output for each trace and secret pair.

    Inputs u0/u1: NTT-domain ciphertext-derived coefficients, each length N.
    candidates: array (M, 2), columns (s0_guess, s1_guess).
    r0 = u0*s0 + zeta*u1*s1 mod 3329
    r1 = u0*s1 + u1*s0 mod 3329
    mul = u0*s0 mod 3329 (ONLY if that product is the real RTL register).
    """
    u0 = np.asarray(u0, dtype=np.int64).reshape(-1)
    u1 = np.asarray(u1, dtype=np.int64).reshape(-1)
    c = np.asarray(candidates)
    if u0.shape != u1.shape:
        raise ValueError("u0 and u1 must have the same shape")
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] == 0:
        raise ValueError("candidates must be an M x 2 array of secret pairs")
    if not np.issubdtype(c.dtype, np.integer):
        raise ValueError("candidate secret coefficients must be integers")
    if np.any(c < 0) or np.any(c >= Q):
        raise ValueError("candidate coefficients must be in 0..3328")
    c = c.astype(np.int64)
    a = u0[:, None] % Q
    b = u1[:, None] % Q
    s0 = c[None, :, 0]
    s1 = c[None, :, 1]
    if component == "r0":
        if zeta is None:
            raise ValueError("r0 requires the correct BaseMul twiddle factor zeta")
        z = _vector(zeta, len(u0), "zeta")[:, None] % Q
        return (a * s0 + z * b * s1) % Q
    if component == "r1":
        return (a * s1 + b * s0) % Q
    if component == "mul":
        return (a * s0) % Q
    raise ValueError("component must be r0, r1, or mul")


class CPA:
    """Vectorized Pearson CPA for one BaseMul coefficient pair/register."""

    def __init__(self, word_bits=16, component="r0", batch_size=64):
        if component not in ("r0", "r1", "mul"):
            raise ValueError("Invalid BaseMul component")
        if not 1 <= word_bits <= 32:
            raise ValueError("word_bits must be in [1, 32]")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.word_bits = word_bits
        self.component = component
        self.batch_size = batch_size

    def attack(self, traces, u0, u1, previous_register, candidates,
               zeta=None, true_pair=None):
        """Return correlation matrix, best candidate, ranking and optional rank.

        previous_register MUST be grounded in a known/captured previous
        register state. No guessed default of zero is used.
        """
        t = np.asarray(traces, dtype=np.float64)
        if t.ndim != 2 or min(t.shape) < 3:
            raise ValueError("traces must be N x T, both dimensions >= 3")
        if not np.isfinite(t).all():
            raise ValueError("traces contain NaN/Inf")
        n, samples = t.shape
        u0 = _vector(u0, n, "ntt_u0")
        u1 = _vector(u1, n, "ntt_u1")
        prev = _vector(previous_register, n, "previous_register")
        if np.any(prev < -(1 << 31)) or np.any(prev >= (1 << 32)):
            raise ValueError("previous register values exceed supported 32-bit width")
        if zeta is not None:
            zeta = _vector(zeta, n, "zeta")
        c = np.asarray(candidates)
        if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] == 0:
            raise ValueError("candidates must be M x 2")
        if not np.issubdtype(c.dtype, np.integer) or np.any(c < 0) or np.any(c >= Q):
            raise ValueError("candidates must contain integers in 0..3328")
        c = c.astype(np.int64)
        centered = t - t.mean(axis=0)
        tnorm = np.linalg.norm(centered, axis=0)
        corr = np.zeros((len(c), samples), dtype=np.float64)
        for start in range(0, len(c), self.batch_size):
            end = min(start + self.batch_size, len(c))
            new = basemul_register_values(
                u0, u1, c[start:end], zeta=zeta, component=self.component)
            leak = hamming_distance(prev[:, None], new, self.word_bits)
            leak -= leak.mean(axis=0)
            lnorm = np.linalg.norm(leak, axis=0)
            denom = lnorm[:, None] * tnorm[None, :]
            corr[start:end] = np.divide(
                leak.T @ centered, denom,
                out=np.zeros((end - start, samples)), where=denom > 0)
        peak = np.max(np.abs(corr), axis=1)
        order = np.argsort(-peak, kind="stable")
        best = int(order[0])
        best_time = int(np.argmax(np.abs(corr[best])))
        result = {
            "correlation_matrix": corr,
            "candidate_pairs": c,
            "peak_correlation": peak,
            "ranking": order,
            "best_pair": tuple(map(int, c[best])),
            "best_score": float(peak[best]),
            "best_sample": best_time,
            "component": self.component,
            "leakage_model": "HD",
        }
        if true_pair is not None:
            actual = np.asarray(true_pair, dtype=np.int64)
            if actual.shape != (2,):
                raise ValueError("true_pair must contain two coefficients")
            matches = np.flatnonzero(np.all(c == actual[None, :], axis=1))
            result["true_pair_rank"] = (
                int(np.flatnonzero(order == matches[0])[0]) + 1
                if matches.size else None
            )
        return result
