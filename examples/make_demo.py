"""Create a fully SYNTHETIC demonstration HDF5 for the HD-CPA GUI.

This does NOT generate valid ML-KEM ciphertexts or genuine FPGA power traces.
The /ciphertexts byte arrays are dummy identifiers, NOT inputs from which
ntt_u0/ntt_u1 are computed. Never use this data as hardware SCA evidence.
"""
from pathlib import Path
import numpy as np
import h5py
from CPA import Q, basemul_register_values, hamming_distance

rng = np.random.default_rng(2026)
n, samples = 360, 160
u0 = rng.integers(0, Q, size=n, dtype=np.int32)
u1 = rng.integers(0, Q, size=n, dtype=np.int32)
prev = rng.integers(0, 65536, size=n, dtype=np.int32)
true_s0, true_s1, zeta = 231, 782, 17
new = basemul_register_values(
    u0, u1, np.array([[true_s0, true_s1]]),
    zeta=zeta, component="r0")[:, 0]
leak = hamming_distance(prev, new, word_bits=16)
traces = rng.normal(0, 0.8, size=(n, samples)).astype(np.float32)
traces[:, 78] += 0.7 * leak.astype(np.float32)
ct_dummy = rng.integers(0, 256, size=(n, 1088), dtype=np.uint8)
output = Path("synthetic_demo.h5")
with h5py.File(output, "w") as f:
    f.create_dataset("traces", data=traces)
    f.create_dataset("ciphertexts", data=ct_dummy)
    f.create_dataset("ntt_u0", data=u0)
    f.create_dataset("ntt_u1", data=u1)
    f.create_dataset("previous_register", data=prev)
    f.attrs["zeta"] = zeta
    f.attrs["synthetic"] = True
    f.attrs["warning"] = "NOT hardware traces; dummy ciphertext identifiers"
print(f"Wrote {output}. Test: --guess s0 --fixed-other {true_s1} "
      f"--true-pair {true_s0} {true_s1}")
