#!/usr/bin/env python3
"""Generate linear algebra fixtures using NumPy/SciPy."""
import pathlib
import numpy as np


def main() -> None:
    fixtures = pathlib.Path("fixtures")
    fixtures.mkdir(exist_ok=True)

    rng = np.random.default_rng(0)

    # Linear solve A x = b (square, well-conditioned)
    A = rng.standard_normal((6, 6)).astype(float)
    # Make A better conditioned by A^T A + I
    A = A.T @ A + np.eye(6)
    b = rng.standard_normal(6).astype(float)
    x = np.linalg.solve(A, b)
    np.save(fixtures / "lin_solve_A.npy", A)
    np.save(fixtures / "lin_solve_b.npy", b)
    np.save(fixtures / "lin_solve_x.npy", x)

    # SVD fixture: rectangular matrix
    As = rng.standard_normal((8, 5)).astype(float)
    np.save(fixtures / "svd_A.npy", As)

    # QR fixture: use a tall matrix
    Aq = rng.standard_normal((7, 5)).astype(float)
    np.save(fixtures / "qr_A.npy", Aq)


if __name__ == "__main__":
    main()

