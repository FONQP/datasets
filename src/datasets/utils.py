import logging
import os
import zipfile

import h5py
import numpy as np
import toml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track

console = Console(record=True)
logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler(console=console, show_time=False)]
)

logger = logging.getLogger(__name__)


def get_freqs(fcen: float, fwidth: float, nfreq: int) -> np.ndarray:
    return np.linspace(fcen - fwidth / 2, fcen + fwidth / 2, nfreq)


def consolidate_h5(root_dir: str, output_path: str) -> None:
    with h5py.File(output_path, "w") as master:
        metadata_toml = os.path.join(root_dir, "metadata.toml")
        if os.path.exists(metadata_toml):
            with open(metadata_toml, "r") as f:
                metadata = toml.load(f)
            for k, v in metadata.items():
                master.attrs[k] = v

        for h5_filename in track(
            os.listdir(os.path.join(root_dir, "data")),
            description="Packing H5 files...",
            console=console,
        ):
            small_file_path = os.path.join(root_dir, "data", h5_filename)

            group_name = f"sample_{os.path.splitext(h5_filename)[0]}"

            try:
                with h5py.File(small_file_path, "r") as src:
                    dest_grp = master.create_group(group_name)

                    for k, v in src.attrs.items():
                        dest_grp.attrs[k] = v

                    for dset_name in src.keys():
                        src.copy(dset_name, dest_grp)

            except Exception as e:
                print(f"Failed to pack {small_file_path}: {e}")


def consolidate_dataset(root_dir: str, output_path: str) -> None:
    for shape_name in os.listdir(root_dir):
        shape_dir = os.path.join(root_dir, shape_name)
        if os.path.isdir(shape_dir):
            output_file = os.path.join(output_path, f"{shape_name}.h5")
            consolidate_h5(shape_dir, output_file)

    output_zip = os.path.join(output_path, "consolidated.zip")
    with zipfile.ZipFile(output_zip, "w") as zipf:
        for shape_name in os.listdir(output_path):
            if shape_name.endswith(".h5"):
                zipf.write(os.path.join(output_path, shape_name), arcname=shape_name)

    print(f"Zipped consolidated files to {output_zip}")


def consolidate_dataset_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate dataset H5 files.")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing shape subdirectories with H5 files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the consolidated H5 file.",
    )

    args = parser.parse_args()
    consolidate_dataset(args.root_dir, args.output_path)
