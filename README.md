# ML-KEM-768 PWM / BaseMul CPA (experimental)

This branch **replaces the original AES S-box byte CPA** with a register-aware
**Hamming Distance (HD) Pearson CPA** prototype for **ML-KEM-768 polynomial
BaseMul / PWM**. It provides a Tkinter GUI, a CLI and regression tests.

**Status: research prototype, not yet validated against Artix-7 RTL traces.**
The core computes *canonical* BaseMul values; real hardware may use
Montgomery/Barrett intermediate representations, signed arithmetic, or a
register storing a different intermediate. Do not claim hardware key recovery
without mapping the predicted value and previous register value to RTL.

## Install

Python 3.10+ recommended.

```sh
python -m pip install -r requirements.txt
python GUI.py
```

Or run without a desktop:

```sh
python Main.py --dataset your_run.h5 --guess s0 --fixed-other 100 \
  --component r0 --start 1000 --stop 2000 --plot
```

`--fixed-other 100` is a **known** other secret coefficient, not a guess
that can be silently assumed. It is useful for an initial known-key lab
experiment. It does **not** represent unconstrained recovery of both BaseMul
secret coefficients.

For an explicitly selected joint candidate list, provide an M-by-2 NumPy
array or CSV:

```sh
python Main.py --dataset your_run.h5 --candidates selected_pairs.npy \
  --component r0 --plot
```

Joint brute force over all pairs would require `3329^2 = 11,082,241`
candidates and may be expensive. The tool ranks **only the supplied
candidate set**.

## Synthetic smoke test (NOT hardware evidence)

```sh
python -m examples.make_demo
python Main.py --dataset synthetic_demo.h5 --guess s0 --fixed-other 782 \\
  --component r0 --true-pair 231 782 --plot
```

Or select `synthetic_demo.h5` in `python GUI.py`, leave Candidate file blank,
set Guess to `s0`, Known other coefficient to `782`, and Component to `r0`.
A visible **SYNTHETIC DEMO** warning identifies the generated data.
The example does not use valid ML-KEM ciphertexts and is not research evidence.

## HDF5 input contract

A single file needs the following datasets (first dimension: same N traces):

| Name | Shape | Meaning |
| --- | --- | --- |
| `/traces` | `(N, T)` | Aligned measured power/EM samples |
| `/ciphertexts` | `(N, 1088)` | Original ML-KEM-768 ciphertext, row aligned with traces |
| `/ntt_u0` | `(N,)` | First NTT-domain public operand for ONE chosen BaseMul pair |
| `/ntt_u1` | `(N,)` | Second NTT-domain public operand for the same pair |
| `/previous_register` | `(N,)` | Actual *previous* register state (known/verified), **not inferred as zero** |
| `/zeta` (optional) | `(N,)` | BaseMul twiddle factor; alternatively root attribute `zeta` or `--zeta` |

**Important:** The HDF5 loader does **not** derive `ntt_u0/u1` from
ciphertext bytes. Precompute those operands with a **verified FIPS 203 /
actual RTL** decode/decompress/NTT pipeline, and preserve the original
ciphertexts for provenance. ML-KEM-768 ciphertexts are 1088 bytes.

For `r0` mode, use the correct zeta for the selected coefficient pair.
The `r1` and `mul` components do not require zeta.

Sampling rate, trigger index, core frequency, bitstream hash and probe
position should be recorded as HDF5 attributes (not yet enforced by the
loader). Crop a physically identified PWM window using `--start` and
`--stop`; these are **zero-based sample positions**, with stop exclusive.

## Leakage model

For each trace i and candidate secret pair (s0, s1), the canonical
ML-KEM BaseMul output candidates are:

```text
r0 = (u0*s0 + zeta*u1*s1) mod 3329
r1 = (u0*s1 + u1*s0) mod 3329
```

An optional `mul = u0*s0 mod 3329` diagnostic mode should be used ONLY
when the targeted RTL register actually stores this product.

```text
L_i(s0,s1) = bit_count(previous_register[i] XOR predicted_register[i])
```

Both values are masked to the configured `--word-bits` (default 16).
The CPA computes Pearson correlation of the hypothesis for every
candidate against every sampled time index in the cropped trace window.
Outputs: **peak absolute correlation**, **candidate rank**, best sample,
and optionally **ground-truth rank** (when known lab secrets are given).

The prior register value must be known/established. If it is itself
candidate-dependent, the simple model here is **not valid**; implement
the full previous-state hypothesis from RTL instead.

Successful ranking within a limited candidate set is **not** full
ML-KEM decapsulation-key recovery, and a CPA correlation peak alone
is not proof of key recovery.

## Test

```sh
python -m unittest discover -s tests -v
```

The tests use synthetic traces with a deliberately injected
HD-correlated signal; they **do not** establish real-device exploitability.

## External code studied (not copied verbatim)

- [anshupriyan/My-CPA-code-analysis](https://github.com/anshupriyan/My-CPA-code-analysis)
  — per-coefficient hypotheses, vectorized correlation, window searches (MIT).
- [Chaman-veteran/KyberCPA](https://github.com/Chaman-veteran/KyberCPA)
  — real capture/known-input association, statistical analysis of Kyber on
  Cortex-M4. ARM instruction models are **not** used for this FPGA prototype.
- [PRASANNA-RAVI/Generic-SCA-CCA-Lattice-Schemes](https://github.com/PRASANNA-RAVI/Generic-SCA-CCA-Lattice-Schemes)
  — chosen ciphertexts, leakage-oriented PoI and clustering/template attacks;
  **not** a drop-in Pearson-CPA implementation.

The original repository implemented **AES-S-box-based CPA** and used
`CPA.py`, `Main.py`, and `ToolBox.py`. Historical waveforms and
`ToolBox.py` remain for reference; the new `Main.py` and `GUI.py` do
not use AES S-box predictions.
