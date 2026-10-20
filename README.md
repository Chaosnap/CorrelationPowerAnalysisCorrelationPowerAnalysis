# ML-KEM-768 Coefficient-Level Correlation Power Analysis

An experimental Python GUI and command-line tool for evaluating a **specific
known-input polynomial multiplier** in an **unprotected ML-KEM FPGA implementation**.
This branch replaces the old **AES-128 S-box / 16-byte** CPA flow.

## Start

Requirements: Python 3.9+ with NumPy, Matplotlib, and a working Tkinter install.

```sh
python -m pip install -r requirements.txt
python -m tkinter             # Verify desktop GUI support
python examples/make_synthetic_demo.py
python Main.py                # GUI: choose synthetic_mlkem_cpa.npz, click Run CPA
python Main.py --cli synthetic_mlkem_cpa.npz --coeff 0 --model HW
python -m unittest discover -s tests
```

The synthetic example should recover coefficient **1234** around sample **43**.
These values are deliberately injected. They **do not** establish FPGA leakage.

## Input format — required

Create **one .npz file** where each row belongs to the same measurement:

| NPZ key | Shape | Interpretation |
| --- | --- | --- |
| `traces` | `(N, T)` | Scope/EM/power captures, one triggered operation per row |
| `operands` | `(N, C)` | **Known coefficient reaching the probed multiplier** for each trace |
| `previous` (optional) | `(N, C)` | Actual previous register value, for HD modeling |

Example conversion from two NumPy arrays:

```python
import numpy as np
# traces: N x samples, input_ntt_coeffs: N x coefficients
assert traces.shape[0] == input_ntt_coeffs.shape[0]
np.savez_compressed("capture_mlkem.npz",
                    traces=traces.astype(np.float32),
                    operands=input_ntt_coeffs.astype(np.int32))
```

**Do not load the old AES demo waveform CSV into this model.** The legacy file
contains AES plaintext/ciphertext fields, not ML-KEM NTT-domain multiplier
operands. UART-received ML-KEM ciphertext bytes **are not automatically the
known `operands`**: decode/decompress and the actual FPGA NTT operations
must be reconstructed for the specific multiplier access.

Trace order must match operand order. Repeated fixed ciphertext with identical
public coefficient values is generally insufficient to distinguish candidate
coefficients using first-order CPA.

## Models and assumptions

Default HW model:

```text
h_i(k) = HW( (u_hat_i * k * factor) mod 3329 )
```

- `k` takes **3329** candidates `0..3328`, appropriate for an **NTT-domain
  secret coefficient**, not necessarily the small original CBD secret.
- `u_hat_i` is the known operand for this **one multiplier operation**.
- `factor=1` assumes the observed intermediate is represented as the
  canonical modular product. If the RTL uses Montgomery multiplication,
  Barrett reduction, signed intermediates, or another register representation,
  derive the correct model from RTL/simulation first.
- The repository of anshupriyan explores `factor=2704` in one script. **That
  numeric factor must not be presumed correct for this FPGA**.
- The **HD** option computes `HW(current XOR previous)`, where `previous`
  must be the observed previous register state, **not the previous ciphertext**.
- These are **single multiplier / reduction-output hypotheses**, NOT the entire
  two-coefficient Kyber/ML-KEM `BaseMul`. The actual BaseMul computes
  cross-products, twiddle factors, additions, and reductions. If the leakage
  is from an accumulator, butterfly, DSP pre-reduction result, or reused
  register, a different hypothesis is required.

## Analysis / GUI

- One target coefficient index (0-based), trace sample window `[start,stop)`.
- Vectorized Pearson correlation over the hypotheses in bounded chunks.
- Maximum absolute correlation over samples and global sample-wise POI profile.
- Top-10 candidates with signed peak correlation, time index and stability
  (number of top-10 appearances over four +/-1..4 sample POI windows).
- Both GUI plots and a headless CLI are available. The GUI does not control
  oscilloscopes, ChipWhisperer, UART, Zynq PS, or an FPGA board.

Results are exploratory; high correlation alone is not a verified key recovery.
Use independent captures, a known ground-truth test vector, target-RTL mapping,
and suitable hold-out data for a defensible hardware SCA result.

## Reference code & provenance

Ideas considered (the new Python implementation here is independent, not
a verbatim import):

- [anshupriyan/My-CPA-code-analysis](https://github.com/anshupriyan/My-CPA-code-analysis):
  `Source_code.py` (coefficient hypotheses, local POI windows),
  `trace_window_for_all.py` (vectorization, optional FFT alignment).
  The latter mixes trace-window selection and circular alignment; neither
  is automatically enabled here.
- [PRASANNA-RAVI/Generic-SCA-CCA-Lattice-Schemes](https://github.com/PRASANNA-RAVI/Generic-SCA-CCA-Lattice-Schemes):
  Kyber512 MATLAB two-class t-test POIs, clustering and reduced templates.
  **Not** a drop-in Pearson CPA implementation.
- [Chaman-veteran/KyberCPA](https://github.com/Chaman-veteran/KyberCPA):
  `statistical_analysis.py` (Pearson over trace samples),
  `get_traces.py` (ChipWhisperer Cortex-M4 acquisition),
  `arm.py` (instruction-specific leakage). ARM intrinsics and
  ChipWhisperer capture APIs should **not** be copied as FPGA leakage models.

The prior AES demonstration and its waveform CSVs are retained in Git history
on the original `master` branch; this branch uses the ML-KEM backend.
