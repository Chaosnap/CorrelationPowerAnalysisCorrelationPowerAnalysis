# CorrelationPowerAnalysis — ML-KEM / Kyber CPA

This repository now uses a coefficient-level Correlation Power Analysis model for
Kyber / ML-KEM instead of the original AES S-box byte model.

## Current leakage model

For a public coefficient `x_j` from trace `j` and a secret coefficient guess
`k in [0, 3328]`, the default PWM/BaseMul model is:

```text
intermediate(j, k) = (x_j * k) mod 3329
leakage(j, k)      = HW(intermediate(j, k))
```

The CPA engine computes Pearson correlation between every hypothesis and every
trace sample, reports the strongest Point of Interest (POI), ranks all 3329
secret-coefficient candidates, and performs a small multi-window stability
check around the global POI.

The default multiplicative-domain factor is `1`, which matches a direct
multiply + Barrett-reduction datapath. A different factor can be supplied for
implementations whose measured intermediate is in another arithmetic domain.

An HD mode is also available when the previous register value is known:

```text
leakage(j, k) = HD(previous_j, intermediate(j, k))
```

## Files

- `CPA.py` — ML-KEM coefficient CPA engine; vectorized Pearson correlation,
  POI detection, top-N candidates and window stability.
- `ToolBox.py` — Hamming Weight / Hamming Distance helpers and optional FFT
  trace alignment.
- `Main.py` — command-line driver for CSV / NPY / single-array NPZ datasets.

## Input format

Use two matrices with the same number of rows:

- **traces**: `n_traces x n_samples`
- **inputs**: `n_traces x n_public_coefficients`

Each input column is the public coefficient associated with one attacked secret
coefficient. Coefficients should normally be represented canonically modulo
3329.

## Example

```bash
python Main.py \
  --traces traces.npy \
  --inputs public_coeffs.csv \
  --coeff 0 \
  --model HW \
  --plot
```

Attack several coefficients:

```bash
python Main.py \
  --traces traces.npy \
  --inputs public_coeffs.csv \
  --coeff 0-9 \
  --num-traces 5000
```

Use the HD model:

```bash
python Main.py \
  --traces traces.npy \
  --inputs public_coeffs.csv \
  --previous previous_register_values.csv \
  --coeff 0 \
  --model HD
```

## Why this replaced the old model

The previous project implemented classic AES CPA:

```text
HW(Sbox(plaintext_byte XOR key_byte))
```

with 256 byte hypotheses. That model is not appropriate for the current
ML-KEM PWM/BaseMul target. ML-KEM uses arithmetic modulo `q = 3329`, so the
attack now evaluates 3329 coefficient hypotheses and models the modular
multiplication intermediate directly.

## Research references used for design comparison

The implementation is original to this repository, but its workflow was
cross-checked against these public research/code resources:

- `anshupriyan/My-CPA-code-analysis` — coefficient-wise Kyber CPA, HW model,
  POI/window analysis and vectorized correlation ideas.
- `PRASANNA-RAVI/Generic-SCA-CCA-Lattice-Schemes` — chosen-ciphertext SCA,
  trace-acquisition synchronization and coefficient-recovery experiment flow.
- `Chaman-veteran/KyberCPA` — Cortex-M4 / ChipWhisperer trace-acquisition and
  host communication workflow.

The first repository is the closest reference for the numerical CPA engine.
The latter two are more useful for acquisition/protocol design than for the
Artix-7 leakage model itself.
