import os
from typing import Dict, List, Optional

import h5py
import meep as mp
import toml
from mpi4py import MPI

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
    comm = MPI.COMM_WORLD

    # Load shape dataset configuration
    shape_config = config["shapes"][shape_name]
    shape_config["name"] = shape_name
    shape_config["material"] = config["shapes"]["material"]
    shape_generator = ShapeGenerator(shape_config)

    # Simulate dataset
    simulator: Optional[Simulator] = None
    shape_config["num_samples"] = (
        config["slurm"]["num_samples_per_job"]
        if slurm_job_id >= 0
        else shape_config["num_samples"]
    )

    iterable = range(shape_config["num_samples"])
    if mp.am_master():
        iterable = track(
            iterable,
            description="[green]Simulating...",
            console=console,
            disable=slurm_job_id >= 0,
        )

    for iter in iterable:
        shape = None
        if mp.am_master():
            shape = shape_generator.get_shape(randomize=True)
            logger.debug(f"Generated shape: {shape.to_dict()}")
        shape = comm.bcast(shape, root=0)

        simulator = Simulator(shape, config)
        simulator.init_simulation_instance(empty=True)
        simulator.run_empty()
        simulator.init_simulation_instance()
        simulator.run()
        if output_dir is not None:
            save_derived_data(
                shape,
                simulator,
                config["measure"],
                output_dir,
                (slurm_job_id * config["slurm"]["num_samples_per_job"]) + iter
                if slurm_job_id >= 0
                else iter,
            )

    if (
        output_dir is not None
        and simulator is not None
        and not slurm_job_id >= 1
        and mp.am_master()
    ):
        metadata = simulator.cell.copy()
        metadata["shape"] = shape_name

        metadata_filename = os.path.join(output_dir, "metadata.toml")
        with open(metadata_filename, "a") as f:
            toml.dump(metadata, f)


def save_derived_data(
    shape: Shape,
    simulator: Simulator,
    measure_list: List[str],
    output_dir: str,
    iter: int,
) -> None:
    results = {}
    if "S_parameters" in measure_list:
        results.update(simulator.get_S_parameters())
    if "EH_fields" in measure_list:
        results.update(simulator.get_EH_fields())
    structure = shape.to_dict()

    filename = os.path.join(output_dir, "data", f"{iter:06d}.h5")

    if mp.am_master():
        with h5py.File(filename, "w") as f:
            for k, v in results.items():
                f.create_dataset(k, data=v, compression="gzip")

            for key, value in structure.items():
                f.attrs[key] = value
