"""Regression tests for canonical ML-KEM BaseMul + register-HD CPA."""
import unittest
import numpy as np
from CPA import CPA, Q, basemul_register_values, hamming_distance


class TestBaseMulHDCPA(unittest.TestCase):
    def test_basemul_pair_formula(self):
        u0 = np.array([3, 5])
        u1 = np.array([7, 11])
        secret = np.array([[13, 17]])
        r0 = basemul_register_values(u0, u1, secret, zeta=19, component="r0")
        r1 = basemul_register_values(u0, u1, secret, component="r1")
        np.testing.assert_array_equal(
            r0[:, 0], (u0 * 13 + 19 * u1 * 17) % Q)
        np.testing.assert_array_equal(
            r1[:, 0], (u0 * 17 + u1 * 13) % Q)

    def test_hamming_distance(self):
        got = hamming_distance(
            np.array([0b1011, 0b0000]),
            np.array([0b0011, 0b1111]), 4)
        np.testing.assert_array_equal(got, [1, 4])
        self.assertEqual(hamming_distance(0xffff, 0, 16), 16)

    def test_recover_known_pair_on_synthetic_traces(self):
        rng = np.random.default_rng(2047)
        n, t = 420, 96
        u0 = rng.integers(0, Q, size=n)
        u1 = rng.integers(0, Q, size=n)
        prev = rng.integers(0, 65536, size=n)
        true_pair = [231, 782]
        candidates = rng.integers(0, Q, size=(80, 2))
        candidates[31] = true_pair
        v = basemul_register_values(
            u0, u1, np.array([true_pair]), zeta=17, component="r0")[:, 0]
        expected_hd = hamming_distance(prev, v, 16)
        traces = rng.normal(0, 0.5, size=(n, t))
        traces[:, 40] += 0.65 * expected_hd
        result = CPA(component="r0", batch_size=19).attack(
            traces, u0, u1, prev, candidates, zeta=17,
            true_pair=true_pair)
        self.assertEqual(result["best_pair"], tuple(true_pair))
        self.assertEqual(result["true_pair_rank"], 1)
        self.assertEqual(result["best_sample"], 40)
        self.assertGreater(result["best_score"], 0.6)

    def test_unknown_previous_state_not_silently_assumed(self):
        rng = np.random.default_rng(9)
        traces = rng.normal(size=(10, 12))
        with self.assertRaises(ValueError):
            CPA().attack(traces, np.arange(10), np.arange(10),
                         None, np.array([[1, 2]]), zeta=3)

    def test_r0_needs_zeta(self):
        with self.assertRaises(ValueError):
            basemul_register_values([1, 2], [1, 2],
                                    [[1, 2]], component="r0")


if __name__ == "__main__":
    unittest.main()
