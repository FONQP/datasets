import os
from argparse import ArgumentParser, Namespace

import h5py
import matplotlib.pyplot as plt
import numpy as np
import toml


def plot():
    parser = ArgumentParser(description="Plot S-parameters")
    parser.add_argument(
        "--csv",
        type=str,
        required=False,
        help="Path to the CSV file containing dataset",
    )
    parser.add_argument(
        "--h5",
        type=str,
        required=False,
        help="Path to the HDF5 file containing S-parameters",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        required=False,
        help="Path to the TOML file containing wavelength configuration",
    )
    args: Namespace = parser.parse_args()

    if args.h5 and args.lambdas:
        plot_h5(args.h5, args.lambdas)
    else:
        data = np.loadtxt(args.csv, delimiter=",")
        if data.shape[1] == 3:
            lambdas = data[:, 0]
            real_part = data[:, 1]
            imag_part = data[:, 2]
            plot_re_im(real_part, imag_part, lambdas)
        elif data.shape[1] == 2:
            if args.lambdas:
                lambdas = np.loadtxt(args.lambdas)
            else:
                lambdas = np.arange(data.shape[0])
            real_part = data[:, 0]
            imag_part = data[:, 1]
            plot_re_im(real_part, imag_part, lambdas)


def plot_meta_atom_idx(dataset, idx):
    real_part = dataset["real"][idx].flatten()
    imag_part = dataset["imag"][idx].flatten()
    plot_re_im(real_part, imag_part, np.arange(len(real_part)))


def plot_re_im(real_part, imag_part, lambdas):
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, real_part, label="Real Part")
    plt.plot(lambdas, imag_part, label="Imaginary Part")
    magnitude = (real_part**2 + imag_part**2) ** 0.5
    plt.plot(lambdas, magnitude, label="Magnitude")
    plt.xlabel("Wavelength Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()


def plot_h5(h5_filepath: str, wavelength_filepath: str) -> None:
    metadata = toml.load(wavelength_filepath)
    fcen = metadata["fcen"]
    fwidth = metadata["fwidth"]
    nfreq = metadata["nfreq"]
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

    plt.plot(lambdas, R, linewidth=2, label="$|S_{11}|^2$ (Reflection)")
    plt.plot(lambdas, T, linewidth=2, label="$|S_{21}|^2$ (Transmission)")

    plt.xlabel(r"Wavelength ($\mu m$)", fontsize=12)
    plt.ylabel("Magnitude", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.show()
