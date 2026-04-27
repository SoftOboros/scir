#!/usr/bin/env python3
"""Generate signal processing fixtures using SciPy."""
import pathlib
import numpy as np
from scipy.signal import butter, cheby1, bessel, iirnotch, sosfilt, sosfilt_zi, resample_poly


def main() -> None:
    fixtures = pathlib.Path("fixtures")
    fixtures.mkdir(exist_ok=True)

    sos_butter = butter(4, 0.2, output="sos")
    sos_cheby = cheby1(4, 1, 0.2, output="sos")
    sos_bessel = bessel(4, 0.2, output="sos")
    # HPF designs exercise the extended *_filter() APIs added in scir-signal 0.3.4
    # (butter, cheby1) and 0.3.5 (bessel).
    sos_butter_hp = butter(4, 0.3, btype="highpass", output="sos")
    sos_cheby_hp = cheby1(4, 1, 0.3, btype="highpass", output="sos")
    sos_bessel_hp = bessel(4, 0.3, btype="highpass", output="sos", norm="phase")
    # iirnotch returns ba; promote to a single-section SOS for parity testing.
    b_notch, a_notch = iirnotch(0.2, 30.0, fs=2.0)
    sos_notch = np.array([[b_notch[0], b_notch[1], b_notch[2], a_notch[0], a_notch[1], a_notch[2]]])

    x = np.linspace(0, 1, 32, endpoint=False)
    y = sosfilt(sos_butter, x)
    y_ff = sosfilt(sos_butter, sosfilt(sos_butter, x)[::-1])[::-1]
    y_cheby = sosfilt(sos_cheby, x)
    y_bessel = sosfilt(sos_bessel, x)
    y_butter_hp = sosfilt(sos_butter_hp, x)
    y_cheby_hp = sosfilt(sos_cheby_hp, x)
    y_bessel_hp = sosfilt(sos_bessel_hp, x)
    y_notch = sosfilt(sos_notch, x)
    y_resample = resample_poly(x, 2, 3)

    np.save(fixtures / "butter_sos.npy", sos_butter)
    np.save(fixtures / "cheby_sos.npy", sos_cheby)
    np.save(fixtures / "bessel_sos.npy", sos_bessel)
    np.save(fixtures / "butter_hp_sos.npy", sos_butter_hp)
    np.save(fixtures / "cheby_hp_sos.npy", sos_cheby_hp)
    np.save(fixtures / "bessel_hp_sos.npy", sos_bessel_hp)
    np.save(fixtures / "notch_sos.npy", sos_notch)
    np.save(fixtures / "sosfilt_input.npy", x)
    np.save(fixtures / "sosfilt_output.npy", y)
    np.save(fixtures / "filtfilt_output.npy", y_ff)
    np.save(fixtures / "cheby_output.npy", y_cheby)
    np.save(fixtures / "bessel_output.npy", y_bessel)
    np.save(fixtures / "butter_hp_output.npy", y_butter_hp)
    np.save(fixtures / "cheby_hp_output.npy", y_cheby_hp)
    np.save(fixtures / "bessel_hp_output.npy", y_bessel_hp)
    np.save(fixtures / "notch_output.npy", y_notch)
    np.save(fixtures / "resample_poly_output.npy", y_resample)


if __name__ == "__main__":
    main()
