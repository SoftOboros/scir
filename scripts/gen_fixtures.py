#!/usr/bin/env python3
"""Generate FFT fixtures of arbitrary size using NumPy/SciPy."""

import argparse
import pathlib
import numpy as np
from scipy.fft import fft, ifft, rfft, irfft


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FFT/IFFT fixtures")
    parser.add_argument(
        "--sizes", "-n", type=int, nargs="+", default=[8], help="FFT sizes to generate"
    )
    args = parser.parse_args()

    fixtures = pathlib.Path("fixtures")
    fixtures.mkdir(exist_ok=True)

    for n in args.sizes:
        data = np.arange(n, dtype=float)
        spectrum = fft(data)
        time = ifft(spectrum)
        r_spectrum = rfft(data)
        r_time = irfft(r_spectrum, n)

        np.save(fixtures / f"fft_input_{n}.npy", data)
        np.save(fixtures / f"fft_output_{n}.npy", spectrum)
        np.save(fixtures / f"ifft_output_{n}.npy", time)
        np.save(fixtures / f"rfft_output_{n}.npy", r_spectrum)
        np.save(fixtures / f"irfft_output_{n}.npy", r_time)


if __name__ == "__main__":
    main()
