import logging
import os

import h5py
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track

console = Console(record=True)
logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler(console=console, show_time=False)]
)

logger = logging.getLogger(__name__)


def consolidate_h5(root_dir: str, output_path: str) -> None:
    with h5py.File(output_path, "w") as master:
        wav_path = os.path.join(root_dir, "wavelengths.npy")
        wavelengths = np.load(wav_path)
        master.create_dataset("wavelengths", data=wavelengths, compression="gzip")

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

    print("Packing complete.")


def consolidate_dataset(root_dir: str, output_path: str) -> None:
    for shape_name in os.listdir(root_dir):
        shape_dir = os.path.join(root_dir, shape_name)
        if os.path.isdir(shape_dir):
            output_file = os.path.join(output_path, f"{shape_name}.h5")
            consolidate_h5(shape_dir, output_file)
