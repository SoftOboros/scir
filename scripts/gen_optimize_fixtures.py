#!/usr/bin/env python3
"""Generate optimization fixtures using SciPy's minimize."""
import pathlib
import numpy as np
from scipy.optimize import minimize


def rosenbrock(xy):
    x, y = xy
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def himmelblau(xy):
    x, y = xy
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def main() -> None:
    fixtures = pathlib.Path("fixtures")
    fixtures.mkdir(exist_ok=True)

    problems = [
        ("rosenbrock", rosenbrock, np.array([-1.2, 1.0])),
        ("himmelblau", himmelblau, np.array([0.0, 0.0])),
    ]

    for name, func, x0 in problems:
        res_nm = minimize(func, x0, method="Nelder-Mead")
        np.save(fixtures / f"{name}_nelder.npy", res_nm.x)
        res_bfgs = minimize(func, x0, method="BFGS")
        np.save(fixtures / f"{name}_bfgs.npy", res_bfgs.x)
        res_lbfgs = minimize(func, x0, method="L-BFGS-B")
        np.save(fixtures / f"{name}_lbfgs.npy", res_lbfgs.x)


if __name__ == "__main__":
    main()
