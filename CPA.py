import numpy as np

from ToolBox import ToolBox


class CPA:
    """Coefficient-level CPA for Kyber / ML-KEM.

    The default leakage model targets the reduced multiplier output used by
    PWM/BaseMul-style arithmetic:

        L = HW((public_coeff * secret_guess * factor) mod q)

    For the current Artix-7 baseline, ``factor`` should normally stay at 1
    because the RTL uses direct multiplication followed by Barrett reduction.
    A different factor can be supplied for Montgomery-domain implementations.
    """

    def __init__(
        self,
        keySize=1,
        q=3329,
        leakage_model="HW",
        montgomery_factor=1,
        top_n=10,
        window_radii=(1, 2, 3, 4),
        guess_chunk=256,
    ):
        self.traceSetCount = 0
        self.key = []
        self.keySize = int(keySize)
        self.q = int(q)
        self.leakage_model = leakage_model.upper()
        self.montgomery_factor = int(montgomery_factor) % self.q
        self.top_n = int(top_n)
        self.window_radii = tuple(int(r) for r in window_radii)
        self.guess_chunk = max(1, int(guess_chunk))

        if self.leakage_model not in {"HW", "HD"}:
            raise ValueError("leakage_model must be 'HW' or 'HD'")

        self.hypothesis = None
        self.plainTexts = None
        self.tracesPoints = None
        self.previousValues = None

        self.matrixRelation = None
        self.matrixRelations = []
        self.results = []

    def SetTracesPointsAndPT(self, tracesPoints, plainTexts):
        """Set traces and public coefficient inputs.

        Parameters
        ----------
        tracesPoints : array-like, shape (n_traces, n_samples)
            Measured power / EM traces.
        plainTexts : array-like, shape (n_traces,) or (n_traces, n_coeffs)
            Public coefficient values associated with each trace. The legacy
            method name is kept so existing callers do not need to change.
        """
        traces = np.asarray(tracesPoints, dtype=np.float64)
        if traces.ndim == 1:
            traces = traces[:, None]
        if traces.ndim != 2:
            raise ValueError("tracesPoints must be a 2-D matrix")

        inputs = self._coerce_inputs(plainTexts)
        if inputs.ndim == 1:
            inputs = inputs[:, None]
        if inputs.ndim != 2:
            raise ValueError("plainTexts/public inputs must be a 1-D or 2-D matrix")

        if traces.shape[0] != inputs.shape[0]:
            raise ValueError(
                f"Trace count mismatch: traces={traces.shape[0]}, inputs={inputs.shape[0]}"
            )

        self.tracesPoints = traces
        self.plainTexts = inputs

    def SetPreviousValues(self, previousValues):
        """Set previous register values for the HD leakage model."""
        values = self._coerce_inputs(previousValues)
        if values.ndim == 1:
            values = values[:, None]
        if self.plainTexts is not None and values.shape[0] != self.plainTexts.shape[0]:
            raise ValueError("previousValues must have the same number of traces as inputs")
        self.previousValues = values

    def GetKey(self):
        return self.key

    def GetResults(self):
        return self.results

    def GetMatrixRelations(self):
        return self.matrixRelations

    def Analyse(self, coeff_indices=None):
        """Run CPA for one or more secret coefficients.

        ``coeff_indices`` selects which public-input columns are attacked.
        When omitted, the first ``keySize`` columns are used.
        """
        if self.tracesPoints is None or self.plainTexts is None:
            raise RuntimeError("Call SetTracesPointsAndPT() before Analyse().")

        n_coeffs = self.plainTexts.shape[1]
        if coeff_indices is None:
            count = min(self.keySize, n_coeffs)
            coeff_indices = list(range(count))
        else:
            coeff_indices = [int(i) for i in coeff_indices]

        for idx in coeff_indices:
            if idx < 0 or idx >= n_coeffs:
                raise IndexError(f"Coefficient index {idx} outside input columns 0..{n_coeffs - 1}")

        self.key = []
        self.matrixRelations = []
        self.results = []

        trace_centered = self.tracesPoints - self.tracesPoints.mean(axis=0, keepdims=True)
        trace_norm = np.linalg.norm(trace_centered, axis=0)

        for coeff_idx in coeff_indices:
            result = self._attack_coefficient(coeff_idx, trace_centered, trace_norm)
            self.key.append(result["best_guess"])
            self.matrixRelation = result["max_corr_per_guess"]
            self.matrixRelations.append(self.matrixRelation)
            self.results.append(result)

        return self.key

    def Correlate(self, A, B):
        """Vectorized Pearson correlation.

        A: (n_observations, n_hypotheses)
        B: (n_observations, n_samples)
        Returns: (n_hypotheses, n_samples)
        """
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)

        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must both be 2-D matrices")
        if A.shape[0] != B.shape[0]:
            raise ValueError("A and B must have the same observation count")

        A_c = A - A.mean(axis=0, keepdims=True)
        B_c = B - B.mean(axis=0, keepdims=True)

        numerator = A_c.T @ B_c
        denominator = (
            np.linalg.norm(A_c, axis=0)[:, None]
            * np.linalg.norm(B_c, axis=0)[None, :]
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = numerator / denominator
        return np.nan_to_num(corr)

    def InitHypothesis(self, coeffNumber=0):
        """Build all q hypotheses for one coefficient.

        Kept for compatibility with the original project. For large datasets,
        Analyse() is preferred because it evaluates hypotheses in chunks.
        """
        if self.plainTexts is None:
            raise RuntimeError("Public coefficient inputs have not been set.")

        coeff_idx = int(coeffNumber)
        if coeff_idx >= self.plainTexts.shape[1] and 1 <= coeff_idx <= self.plainTexts.shape[1]:
            coeff_idx -= 1

        keys = np.arange(self.q, dtype=np.int64)
        self.hypothesis = self._build_hypotheses(coeff_idx, keys)
        return self.hypothesis

    def _attack_coefficient(self, coeff_idx, trace_centered, trace_norm):
        n_samples = trace_centered.shape[1]
        max_corr_per_guess = np.zeros(self.q, dtype=np.float64)
        peak_sample_per_guess = np.zeros(self.q, dtype=np.int64)
        poi_curve = np.zeros(n_samples, dtype=np.float64)

        for start in range(0, self.q, self.guess_chunk):
            stop = min(start + self.guess_chunk, self.q)
            keys = np.arange(start, stop, dtype=np.int64)

            h = self._build_hypotheses(coeff_idx, keys)
            h_centered = h - h.mean(axis=0, keepdims=True)
            h_norm = np.linalg.norm(h_centered, axis=0)

            numerator = h_centered.T @ trace_centered
            denominator = h_norm[:, None] * trace_norm[None, :]

            with np.errstate(divide="ignore", invalid="ignore"):
                corr = numerator / denominator
            corr = np.nan_to_num(corr)
            abs_corr = np.abs(corr)

            local_pos = np.argmax(abs_corr, axis=1)
            rows = np.arange(stop - start)
            max_corr_per_guess[start:stop] = corr[rows, local_pos]
            peak_sample_per_guess[start:stop] = local_pos
            poi_curve = np.maximum(poi_curve, abs_corr.max(axis=0))

        best_guess = int(np.argmax(np.abs(max_corr_per_guess)))
        best_peak_sample = int(peak_sample_per_guess[best_guess])
        best_peak_corr = float(max_corr_per_guess[best_guess])

        global_poi = int(np.argmax(poi_curve))
        windows = []
        for radius in self.window_radii:
            lo = max(0, global_poi - radius)
            hi = min(n_samples, global_poi + radius + 1)
            windows.append((lo, hi))

        window_scores = np.zeros((len(windows), self.q), dtype=np.float64)

        for start in range(0, self.q, self.guess_chunk):
            stop = min(start + self.guess_chunk, self.q)
            keys = np.arange(start, stop, dtype=np.int64)

            h = self._build_hypotheses(coeff_idx, keys)
            h_centered = h - h.mean(axis=0, keepdims=True)
            h_norm = np.linalg.norm(h_centered, axis=0)

            numerator = h_centered.T @ trace_centered
            denominator = h_norm[:, None] * trace_norm[None, :]

            with np.errstate(divide="ignore", invalid="ignore"):
                corr = numerator / denominator
            corr = np.nan_to_num(corr)

            for w_idx, (lo, hi) in enumerate(windows):
                window_scores[w_idx, start:stop] = np.sum(corr[:, lo:hi] ** 2, axis=1)

        top_n = min(self.top_n, self.q)
        top_keys_by_window = []
        stability = np.zeros(self.q, dtype=np.int64)

        for scores in window_scores:
            top_keys = np.argsort(scores)[-top_n:][::-1]
            top_keys_by_window.append(top_keys.tolist())
            stability[top_keys] += 1

        ranked_by_peak = np.argsort(np.abs(max_corr_per_guess))[-top_n:][::-1]

        return {
            "coeff": int(coeff_idx),
            "best_guess": best_guess,
            "peak_sample": best_peak_sample,
            "peak_corr": best_peak_corr,
            "global_poi": global_poi,
            "global_poi_corr": float(poi_curve[global_poi]),
            "top_guesses": [
                {
                    "guess": int(k),
                    "corr": float(max_corr_per_guess[k]),
                    "sample": int(peak_sample_per_guess[k]),
                    "stability": int(stability[k]),
                }
                for k in ranked_by_peak
            ],
            "windows": windows,
            "top_keys_by_window": top_keys_by_window,
            "stability": stability,
            "poi_curve": poi_curve,
            "max_corr_per_guess": max_corr_per_guess,
            "peak_sample_per_guess": peak_sample_per_guess,
        }

    def _build_hypotheses(self, coeff_idx, keys):
        x = np.asarray(self.plainTexts[:, coeff_idx], dtype=np.int64) % self.q
        keys = np.asarray(keys, dtype=np.int64) % self.q

        intermediate = (
            x[:, None]
            * keys[None, :]
            * np.int64(self.montgomery_factor)
        ) % self.q

        if self.leakage_model == "HW":
            return ToolBox.HammingWeightArray(intermediate, bit_width=12).astype(np.float64)

        if self.previousValues is None:
            raise RuntimeError(
                "HD model requires SetPreviousValues() with the previous register value."
            )

        prev_col = min(coeff_idx, self.previousValues.shape[1] - 1)
        prev = np.asarray(self.previousValues[:, prev_col], dtype=np.int64) % self.q
        return ToolBox.HammingDistanceArray(
            prev[:, None], intermediate, bit_width=12
        ).astype(np.float64)

    @staticmethod
    def _coerce_inputs(values):
        arr = np.asarray(values)

        if arr.dtype.kind in {"U", "S", "O"}:
            def parse(v):
                if isinstance(v, bytes):
                    v = v.decode()
                if isinstance(v, str):
                    s = v.strip()
                    try:
                        return int(s, 0)
                    except ValueError:
                        return int(s, 16)
                return int(v)

            arr = np.vectorize(parse, otypes=[np.int64])(arr)

        return np.asarray(arr, dtype=np.int64)
