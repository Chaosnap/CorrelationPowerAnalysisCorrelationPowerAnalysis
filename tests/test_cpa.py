"""Deterministic, synthetic-only checks for the new ML-KEM CPA engine."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from CPA import MLKEMCPA, Q, _popcount


class CoefficientCPATests(unittest.TestCase):
    def make_data(self, mode="HW", n=220, t=48, secret=1234):
        rng = np.random.default_rng(2026)
        x = rng.integers(0, Q, size=n, dtype=np.int64)
        previous = rng.integers(0, 65536, size=n, dtype=np.int64)
        intermediate = (x * secret) % Q
        if mode == "HW":
            leakage = _popcount(intermediate)
        else:
            leakage = _popcount(intermediate ^ previous)
        traces = rng.normal(0, 0.5, size=(n, t))
        traces[:, 17] += 2.5 * leakage
        return traces, x.reshape(-1, 1), previous.reshape(-1, 1)

    def test_hw_recovers_known_coefficient(self):
        t, o, p = self.make_data()
        r = MLKEMCPA().analyze(t, o, coefficient=0, start=5, stop=35)
        self.assertEqual(r["best_key"], 1234)
        self.assertEqual(r["peak_sample"], 17)
        self.assertGreater(abs(r["peak_corr"]), 0.8)

    def test_hd_recovers_with_known_previous_state(self):
        t, o, p = self.make_data(mode="HD")
        r = MLKEMCPA().analyze(t, o, previous=p, model="HD")
        self.assertEqual(r["best_key"], 1234)
        self.assertEqual(r["peak_sample"], 17)

    def test_hd_requires_previous(self):
        t, o, p = self.make_data()
        with self.assertRaisesRegex(ValueError, "previous"):
            MLKEMCPA().analyze(t, o, model="HD")

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            MLKEMCPA().analyze(np.ones((5, 10)), np.ones((4, 1), dtype=int))


if __name__ == "__main__":
    unittest.main()
