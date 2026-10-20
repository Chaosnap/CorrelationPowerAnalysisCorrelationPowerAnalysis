"""Generate a SYNTHETIC CPA dataset. Not an FPGA capture or security result."""
import numpy as np

from pathlib import Path

Q = 3329
RNG = np.random.default_rng(2026)
N, T = 300, 100
KNOWN_SECRET_NTT_COEFFICIENT = 1234
operand = RNG.integers(0, Q, size=N, dtype=np.int64)
intermediate = (operand * KNOWN_SECRET_NTT_COEFFICIENT) % Q
hw = np.array([bin(int(x)).count("1") for x in intermediate])
traces = RNG.normal(0, 1, size=(N, T)).astype(np.float32)
traces[:, 43] += (3.0 * hw).astype(np.float32)
outfile = Path(__file__).resolve().parents[1] / "synthetic_mlkem_cpa.npz"
np.savez_compressed(outfile, traces=traces,
                    operands=operand[:, None].astype(np.int32))
print("Saved:", outfile)
print("Synthetic test only. Expected coefficient:", KNOWN_SECRET_NTT_COEFFICIENT)
