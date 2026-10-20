import numpy as np


class ToolBox:
    """Small leakage-model helpers for Kyber / ML-KEM."""

    _HW8 = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)

    def __init__(self):
        pass

    @classmethod
    def HammingWeight(cls, num, ret=1):
        """Return the ordinary Hamming weight of an integer.

        ``ret`` is kept only for compatibility with the original project.
        """
        value = int(num)
        if value < 0:
            value &= 0xFFFFFFFF
        return value.bit_count()

    @classmethod
    def HammingDistance(cls, num1, num2):
        return (int(num1) ^ int(num2)).bit_count()

    @classmethod
    def HammingWeightArray(cls, values, bit_width=32):
        x = np.asarray(values, dtype=np.uint64)
        width = max(1, int(bit_width))
        result = np.zeros(x.shape, dtype=np.uint16)

        for shift in range(0, width, 8):
            result += cls._HW8[((x >> shift) & 0xFF).astype(np.uint8)]

        return result

    @classmethod
    def HammingDistanceArray(cls, values_a, values_b, bit_width=32):
        a = np.asarray(values_a, dtype=np.uint64)
        b = np.asarray(values_b, dtype=np.uint64)
        return cls.HammingWeightArray(np.bitwise_xor(a, b), bit_width=bit_width)

    @staticmethod
    def ModQ(values, q=3329):
        return np.asarray(values, dtype=np.int64) % int(q)

    @staticmethod
    def AlignTracesFFT(traces):
        """Align traces to the first trace using FFT cross-correlation."""
        traces = np.asarray(traces, dtype=np.float64)
        if traces.ndim != 2 or len(traces) == 0:
            return traces

        reference = traces[0]
        ref_fft = np.fft.fft(reference)
        aligned = np.empty_like(traces)

        for i, trace in enumerate(traces):
            cross_corr = np.fft.ifft(np.fft.fft(trace) * np.conj(ref_fft))
            shift = int(np.argmax(np.abs(cross_corr)))
            if shift > trace.size // 2:
                shift -= trace.size
            aligned[i] = np.roll(trace, -shift)

        return aligned
