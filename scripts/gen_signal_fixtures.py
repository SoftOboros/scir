#!/usr/bin/env python3
"""Generate signal processing fixtures using SciPy."""
import pathlib
import numpy as np
from scipy.signal import butter, cheby1, bessel, sosfilt, resample_poly


def main() -> None:
    fixtures = pathlib.Path("fixtures")
    fixtures.mkdir(exist_ok=True)

    sos_butter = butter(4, 0.2, output="sos")
    sos_cheby = cheby1(4, 1, 0.2, output="sos")
    sos_bessel = bessel(4, 0.2, output="sos")

    x = np.linspace(0, 1, 32, endpoint=False)
    y = sosfilt(sos_butter, x)
    y_ff = sosfilt(sos_butter, sosfilt(sos_butter, x)[::-1])[::-1]
    y_cheby = sosfilt(sos_cheby, x)
    y_bessel = sosfilt(sos_bessel, x)
    y_resample = resample_poly(x, 2, 3)

    np.save(fixtures / "butter_sos.npy", sos_butter)
    np.save(fixtures / "cheby_sos.npy", sos_cheby)
    np.save(fixtures / "bessel_sos.npy", sos_bessel)
    np.save(fixtures / "sosfilt_input.npy", x)
    np.save(fixtures / "sosfilt_output.npy", y)
    np.save(fixtures / "filtfilt_output.npy", y_ff)
    np.save(fixtures / "cheby_output.npy", y_cheby)
    np.save(fixtures / "bessel_output.npy", y_bessel)
    np.save(fixtures / "resample_poly_output.npy", y_resample)


if __name__ == "__main__":
    main()
