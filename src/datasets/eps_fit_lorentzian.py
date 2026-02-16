import sys
from typing import Tuple

import matplotlib.pyplot as plt
import meep as mp
import nlopt
import numpy as np


def lorentzfunc(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Returns the complex ε profile given a set of Lorentzian parameters p
    (σ_0, ω_0, γ_0, σ_1, ω_1, γ_1, ...) for a set of frequencies x.
    """
    N = len(p) // 3
    y = np.zeros(len(x))
    for n in range(N):
        A_n = p[3 * n + 0]
        x_n = p[3 * n + 1]
        g_n = p[3 * n + 2]
        y = y + A_n / (np.square(x_n) - np.square(x) - 1j * x * g_n)
    return y


def lorentzerr(p: np.ndarray, x: np.ndarray, y: np.ndarray, grad: np.ndarray) -> float:
    """
    Returns the error (L2 norm) between ε(p,x) and y.
    Also calculates the gradient for the optimizer.
    """
    N = len(p) // 3
    yp = lorentzfunc(p, x)
    val = np.sum(np.square(abs(y - yp)))
    for n in range(N):
        A_n = p[3 * n + 0]
        x_n = p[3 * n + 1]
        g_n = p[3 * n + 2]
        d = 1 / (np.square(x_n) - np.square(x) - 1j * x * g_n)
        if grad.size > 0:
            grad[3 * n + 0] = 2 * np.real(np.dot(np.conj(yp - y), d))
            grad[3 * n + 1] = (
                -4 * x_n * A_n * np.real(np.dot(np.conj(yp - y), np.square(d)))
            )
            grad[3 * n + 2] = (
                -2 * A_n * np.imag(np.dot(np.conj(yp - y), x * np.square(d)))
            )
    return val


def lorentzfit(
    p0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    alg=nlopt.LD_LBFGS,
    tol: float = 1e-25,
    maxeval: float = 10000,
) -> Tuple[np.ndarray, float]:
    """
    Runs the NLopt optimization to find the best lorentzian parameters.
    """
    opt = nlopt.opt(alg, len(p0))
    opt.set_ftol_rel(tol)
    opt.set_maxeval(maxeval)
    opt.set_lower_bounds(np.zeros(len(p0)))
    opt.set_upper_bounds(float("inf") * np.ones(len(p0)))
    opt.set_min_objective(lambda p, grad: lorentzerr(p, x, y, grad))
    local_opt = nlopt.opt(nlopt.LD_LBFGS, len(p0))
    local_opt.set_ftol_rel(1e-10)
    local_opt.set_xtol_rel(1e-8)
    opt.set_local_optimizer(local_opt)
    popt = opt.optimize(p0)
    minf = opt.last_optimum_value()
    return popt, minf


def load_refractive_index_info(filename):
    """
    Parses a refractiveindex.info YAML-style text file.
    Extracts columns: wavelength(um), n, k
    """
    data = []
    reading_data = False

    with open(filename, "r") as f:
        for line in f:
            stripped = line.strip()
            if "data: |" in stripped:
                reading_data = True
                continue

            if reading_data and (":" in stripped or stripped == ""):
                try:
                    float(stripped.split()[0])
                except (ValueError, IndexError):
                    if len(data) > 0:
                        reading_data = False

            if reading_data:
                try:
                    parts = list(map(float, stripped.split()))
                    if len(parts) >= 3:
                        # Format: wavelength, n, k
                        data.append(parts[:3])
                except ValueError:
                    continue

    return np.array(data)


if __name__ == "__main__":
    DATA_FILE = sys.argv[1]
    EPS_INF = 1.1  # Instantaneous dielectric factor (>1 for stability)
    NUM_LORENTZIANS = 3  # Increase if fit is poor
    NUM_REPEAT = 30  # Number of random restarts to avoid local minima

    # Range of wavelengths to fit (in microns)
    # Adjust these to match the range of interest in your simulation
    WL_MIN_UM = float(sys.argv[2])
    WL_MAX_UM = float(sys.argv[3])

    print(f"Loading data from {DATA_FILE}...")
    raw_data = load_refractive_index_info(DATA_FILE)

    if len(raw_data) == 0:
        raise ValueError("No data found! Check the file format.")

    wl_um = raw_data[:, 0]
    n_val = raw_data[:, 1]
    k_val = raw_data[:, 2]

    n_complex = n_val + 1j * k_val
    eps_actual = np.square(n_complex) - EPS_INF

    mask = (wl_um >= WL_MIN_UM) & (wl_um <= WL_MAX_UM)
    wl_reduced = wl_um[mask]
    eps_reduced = eps_actual[mask]

    # Convert Wavelength (um) to Frequency (1/um)
    # Meep unit frequency = c/a. If a=1um, f = 1/wavelength_um
    freqs_reduced = 1.0 / wl_reduced

    print(
        f"Fitting {len(wl_reduced)} data points using {NUM_LORENTZIANS} Lorentzian terms..."
    )

    ps = np.zeros((NUM_REPEAT, 3 * NUM_LORENTZIANS))
    mins = np.zeros(NUM_REPEAT)

    for m in range(NUM_REPEAT):
        p_rand = [10 ** (np.random.random()) for _ in range(3 * NUM_LORENTZIANS)]
        try:
            ps[m, :], mins[m] = lorentzfit(
                p_rand,  # type: ignore
                freqs_reduced,
                eps_reduced,
                nlopt.LD_MMA,
                1e-25,
                50000,
            )
        except Exception as e:
            print(f"Iteration {m} failed: {e}")
            mins[m] = float("inf")

    idx_opt = np.argmin(mins)
    best_p = ps[idx_opt]
    print(f"Best fit error: {mins[idx_opt]:.6f}")

    susceptibilities_code = []

    print("\n--- COPY THIS BLOCK INTO YOUR MEEP SCRIPT ---\n")
    print("E_susceptibilities = [")

    for n in range(NUM_LORENTZIANS):
        sigma = best_p[3 * n + 0]
        frq = best_p[3 * n + 1]
        gam = best_p[3 * n + 2]

        if frq == 0:
            # Drude term
            print(
                f"    mp.DrudeSusceptibility(frequency=1.0, gamma={gam:.5f}, sigma={sigma:.5f}),"
            )
        else:
            # Lorentzian term
            # Convert sigma parameter for Meep definition
            sigma_meep = sigma / (frq**2)
            print(
                f"    mp.LorentzianSusceptibility(frequency={frq:.5f}, gamma={gam:.5f}, sigma={sigma_meep:.5f}),"
            )

    print("]")
    print(
        f"my_material = mp.Medium(epsilon={EPS_INF}, E_susceptibilities=E_susceptibilities, valid_freq_range=mp.FreqRange(min={1 / WL_MAX_UM:.5f}, max={1 / WL_MIN_UM:.5f}))"
    )
    print("\n---------------------------------------------\n")

    E_susceptibilities = []
    for n in range(NUM_LORENTZIANS):
        frq = best_p[3 * n + 1]
        gam = best_p[3 * n + 2]
        if frq == 0:
            sigma = best_p[3 * n + 0]
            E_susceptibilities.append(
                mp.DrudeSusceptibility(frequency=1.0, gamma=gam, sigma=sigma)
            )
        else:
            sigma = best_p[3 * n + 0] / frq**2
            E_susceptibilities.append(
                mp.LorentzianSusceptibility(frequency=frq, gamma=gam, sigma=sigma)
            )

    dummy_medium = mp.Medium(epsilon=EPS_INF, E_susceptibilities=E_susceptibilities)

    eps_fit = np.array([dummy_medium.epsilon(f)[0][0] for f in freqs_reduced])

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(wl_reduced, np.real(eps_reduced) + EPS_INF, "b-", lw=2, label="Data")
    ax[0].plot(wl_reduced, np.real(eps_fit), "r--", lw=2, label="Fit")
    ax[0].set_title(r"Real $\varepsilon$")
    ax[0].set_xlabel(r"Wavelength ($\mu m$)")
    ax[0].legend()

    ax[1].plot(wl_reduced, np.imag(eps_reduced), "b-", lw=2, label="Data")
    ax[1].plot(wl_reduced, np.imag(eps_fit), "r--", lw=2, label="Fit")
    ax[1].set_title(r"Imag $\varepsilon$")
    ax[1].set_xlabel(r"Wavelength ($\mu m$)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
