import os
from typing import Dict, List, Optional

import h5py
import meep as mp
import toml

from datasets.shapes import Shape, ShapeGenerator
from datasets.simulator import Simulator
from datasets.utils import console, logger, track


def simulate_shape(
    shape_name: str,
    config: Dict,
    output_dir: Optional[str] = None,
    slurm_job_id: int = -1,
) -> None:
    mp.verbosity(0)

    # Load shape dataset configuration
    shape_config = config["shapes"][shape_name]
    shape_config["name"] = shape_name
    shape_config["material"] = config["shapes"]["material"]
    shape_generator = ShapeGenerator(shape_config)

    # Simulate dataset
    simulator: Optional[Simulator] = None
    shape_config["num_samples"] = 1 if slurm_job_id >= 0 else shape_config["num_samples"]
    for iter in track(
        range(shape_config["num_samples"]),
        description=f"[green]Simulating {shape_name} shapes...",
        console=console,
        disable=slurm_job_id >= 0,
    ):
        shape = shape_generator.get_shape(randomize=True)
        logger.debug(f"Generated shape: {shape.to_dict()}")

        simulator = Simulator(shape, config)
        simulator.init_simulation_instance(empty=True)
        simulator.run_empty()
        simulator.init_simulation_instance()
        simulator.run()
        if output_dir is not None:
            save_derived_data(shape, simulator, config["get"], output_dir, slurm_job_id if slurm_job_id >= 0 else iter)

    if output_dir is not None and simulator is not None and not slurm_job_id >= 0:
        metadata = simulator.get_sim_cell_generics()
        metadata["shape"] = shape_name

        metadata_filename = os.path.join(output_dir, "metadata.toml")
        with open(metadata_filename, "a") as f:
            toml.dump({shape_name: metadata}, f)


def save_derived_data(
    shape: Shape, simulator: Simulator, save_list: List[str], output_dir: str, iter: int
) -> None:
    structure = shape.to_dict()

    S_parameters = simulator.get_S_parameters() if "S_parameters" in save_list else {}

    filename = f"{output_dir}/data/{iter:06d}.h5"
    if mp.am_master():
        with h5py.File(filename, "w") as f:
            if "S_parameters" in save_list:
                for k, v in S_parameters.items():
                    if k in ["S11", "S21"]:
                        f.create_dataset(k, data=v, compression="gzip")

            for key, value in structure.items():
                f.attrs[key] = value
