import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import toml


def plot_meta_atom_idx(dataset, idx):
    real_part = dataset["real"][idx].flatten()
    imag_part = dataset["imag"][idx].flatten()
    plot_re_im(real_part, imag_part)


def plot_re_im(real_part, imag_part):
    plt.figure(figsize=(10, 6))
    plt.plot(real_part, label="Real Part")
    plt.plot(imag_part, label="Imaginary Part")
    magnitude = (real_part**2 + imag_part**2) ** 0.5
    plt.plot(magnitude, label="Magnitude")
    plt.xlabel("Frequency Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()


def plot_h5(h5_filepath: str, wavelength_filepath: str) -> None:
    toml_data = toml.load(wavelength_filepath)
    fcen = toml_data["fcen"]
    fwidth = toml_data["fwidth"]
    nfreq = toml_data["nfreq"]
    frequencies = np.linspace(fcen - fwidth / 2, fcen + fwidth / 2, nfreq)
    wavelengths = 1 / frequencies

    with h5py.File(h5_filepath, "r") as f:
        s11_complex = f["S11"][:]  # type: ignore
        s21_complex = f["S21"][:]  # type: ignore
        shape_name = f.attrs.get("name", "Unknown Shape")

    title = f"S Parameters for {shape_name}: {os.path.basename(h5_filepath)}"
    plot_s11_s21(s11_complex, s21_complex, wavelengths, title)


def plot_s11_s21(
    s11: np.ndarray, s21: np.ndarray, lambdas: np.ndarray, title: str = ""
) -> None:
    R = np.abs(s11) ** 2
    T = np.abs(s21) ** 2

    plt.figure(figsize=(10, 6))

    plt.plot(lambdas, R, "r-", linewidth=2, label="$|S_{11}|^2$ (Reflection)")
    plt.plot(lambdas, T, "b-", linewidth=2, label="$|S_{21}|^2$ (Transmission)")

    plt.xlabel(r"Wavelength ($\mu m$)", fontsize=12)
    plt.ylabel("Magnitude", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.show()
