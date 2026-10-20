"""Tests for the HDF5-to-CPA path, including data provenance constraints."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from CPA import basemul_register_values, hamming_distance
from Main import analyze_file, load_dataset


class TestHDF5Integration(unittest.TestCase):
    def _write_dataset(self, filename, previous=True):
        rng = np.random.default_rng(321)
        n, samples = 240, 60
        u0 = rng.integers(0, 3329, size=n, dtype=np.int32)
        u1 = rng.integers(0, 3329, size=n, dtype=np.int32)
        prev = rng.integers(0, 65536, size=n, dtype=np.int32)
        pair = np.array([[17, 29]])
        new = basemul_register_values(u0, u1, pair, zeta=7)[:, 0]
        traces = rng.normal(0, 0.5, size=(n, samples))
        traces[:, 20] += hamming_distance(prev, new, 16) * 0.9
        with h5py.File(filename, "w") as f:
            f.attrs["zeta"] = 7
            f.attrs["synthetic"] = True
            f.create_dataset("traces", data=traces)
            f.create_dataset("ciphertexts", data=np.zeros((n, 1088), dtype=np.uint8))
            f.create_dataset("ntt_u0", data=u0)
            f.create_dataset("ntt_u1", data=u1)
            if previous:
                f.create_dataset("previous_register", data=prev)

    def test_pipeline_recovers_pair_and_labels_synthetic(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "pilot.h5"
            self._write_dataset(p)
            pairs = np.array([[17, 29], [11, 23], [19, 41], [0, 3]])
            candidates = Path(temp) / "pairs.npy"
            np.save(candidates, pairs)
            result = analyze_file(
                dataset=str(p), candidates_path=str(candidates),
                component="r0", start=10, stop=35,
                true_pair=[17, 29])
            self.assertEqual(result["best_pair"], (17, 29))
            self.assertEqual(result["best_sample_absolute"], 20)
            self.assertEqual(result["true_pair_rank"], 1)
            self.assertTrue(result["synthetic"])

    def test_rejects_missing_previous_register(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "missing.h5"
            self._write_dataset(p, previous=False)
            with self.assertRaisesRegex(ValueError, "previous_register"):
                load_dataset(p)


if __name__ == "__main__":
    unittest.main()
