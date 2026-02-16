import os
from argparse import ArgumentParser, Namespace
from typing import Dict, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import toml

from datasets.utils import get_freqs


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
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the generated plot",
    )
    parser.add_argument(
        "--attr",
        type=bool,
        default=True,
        help="Prints HDF5 attributes if set to True",
    )
    args: Namespace = parser.parse_args()

    if args.h5 and args.lambdas:
        plot_h5(args.h5, args.lambdas, save_path=args.save)
        if args.attr:
            with h5py.File(args.h5, "r") as f:
                print("HDF5 Attributes:")
                for key, value in f.attrs.items():
                    print(f"{key}: {value}")
    else:
        data = np.loadtxt(args.csv, delimiter=",")
        if data.shape[1] == 3:
            lambdas = data[:, 0]
            real_part = data[:, 1]
            imag_part = data[:, 2]
            plot_re_im(real_part, imag_part, lambdas, save_path=args.save)
        elif data.shape[1] == 2:
            if args.lambdas:
                lambdas = np.loadtxt(args.lambdas)
            else:
                lambdas = np.arange(data.shape[0])
            real_part = data[:, 0]
            imag_part = data[:, 1]
            plot_re_im(real_part, imag_part, lambdas, save_path=args.save)


def plot_meta_atom_idx(dataset, idx):
    real_part = dataset["real"][idx].flatten()
    imag_part = dataset["imag"][idx].flatten()
    plot_re_im(real_part, imag_part, np.arange(len(real_part)))


def plot_re_im(real_part, imag_part, lambdas, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, real_part, label="Real Part")
    plt.plot(lambdas, imag_part, label="Imaginary Part")
    magnitude = (real_part**2 + imag_part**2) ** 0.5
    plt.plot(lambdas, magnitude, label="Magnitude")
    plt.xlabel("Wavelength Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_h5(
    h5_filepath: str, wavelength_filepath: str, save_path: Optional[str] = None
) -> None:
    metadata = toml.load(wavelength_filepath)
    wavelengths = 1 / get_freqs(metadata["fcen"], metadata["fwidth"], metadata["nfreq"])
    dft_wavelengths = 1 / get_freqs(
        metadata["fcen"],
        metadata["fwidth"],
        metadata.get("dft_nfreqs", metadata["nfreq"]),
    )

    with h5py.File(h5_filepath, "r") as f:
        s11_complex = f["S11"][:]  # type: ignore
        s21_complex = f["S21"][:]  # type: ignore
        ex = f["Ex"][:] if "Ex" in f else None  # type: ignore
        ey = f["Ey"][:] if "Ey" in f else None  # type: ignore
        ez = f["Ez"][:] if "Ez" in f else None  # type: ignore
        hx = f["Hx"][:] if "Hx" in f else None  # type: ignore
        hy = f["Hy"][:] if "Hy" in f else None  # type: ignore
        hz = f["Hz"][:] if "Hz" in f else None  # type: ignore
        shape_name = f.attrs.get("name", "Unknown Shape")

    title = f"S Parameters for {shape_name}: {os.path.basename(h5_filepath)}"
    plot_S_parameters(
        s11_complex,  # type: ignore
        s21_complex,  # type: ignore
        wavelengths,
        title,
        save_path=f"{save_path}_s11_s21.png",
    )

    if ex is not None and ey is not None and ez is not None:
        fields_dict = {
            "Ex": ex,
            "Ey": ey,
            "Ez": ez,
            "Hx": hx,
            "Hy": hy,
            "Hz": hz,
        }
        fields_dict = {k: v for k, v in fields_dict.items() if v is not None}
        plot_EH_fields(
            fields_dict, dft_wavelengths, save_path=f"{save_path}_eh_fields.png"
        )


def plot_S_parameters(
    s11: np.ndarray,
    s21: np.ndarray,
    lambdas: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
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

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_EH_fields(
    fields: Dict[str, np.ndarray],
    lambdas: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    num_rows = len(lambdas)
    num_fields = len(fields)
    plt.figure(figsize=(6 * num_fields, 5 * num_rows))
    field_names = list(fields.keys())
    for i in range(num_rows):
        for j, component in enumerate(field_names):
            plt.subplot(num_rows, num_fields, i * num_fields + j + 1)
            data = np.asarray(fields[component])
            plt.imshow(
                np.abs(data[i].T),
                origin="lower",
                extent=(0, data.shape[2], 0, data.shape[1]),
                cmap="viridis",
            )
            plt.colorbar(label=f"|{component}|", shrink=0.8)
            plt.xlabel("x (μm)", fontsize=26)
            plt.ylabel("y (μm)", fontsize=26)
            plt.title(f"{component} at λ={lambdas[i]:.2f} μm", fontsize=30)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
