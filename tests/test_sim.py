import os

import matplotlib.pyplot as plt
import meep as mp
import toml

from datasets.plot import plot_EH_fields, plot_S_parameters
from datasets.shapes import ShapeGenerator
from datasets.simulator import Simulator
from datasets.utils import get_freqs


def test_sim_1():
    config = toml.load(
        f"{os.path.dirname(os.path.abspath(__file__))}/../configs/test_config.toml"
    )
    shape_config = config["shapes"]["polygon"]
    shape_config["name"] = "polygon"
    shape_config["material"] = config["shapes"]["material"]
    shape_generator = ShapeGenerator(shape_config)
    shape = shape_generator.get_shape(randomize=False)

    simulator = Simulator(shape, config)
    simulator.init_simulation_instance(empty=True)
    simulator.run_empty()
    simulator.init_simulation_instance()
    simulator.run()

    # Plot cell slice
    cell = simulator.cell["cell_size"]
    vertical_slice = mp.Volume(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(cell.x, 0, cell.z),
        dims=2,
    )

    simulator.sim.plot2D(
        output_plane=vertical_slice,
        eps_parameters={"cmap": "binary", "interpolation": "none"},
        show_sources=True,
        show_monitors=True,
    )
    plt.savefig("test_simulation_plot.png")

    # Plot S-parameters
    S_parameters = simulator.get_S_parameters()
    wavelengths = 1 / get_freqs(
        simulator.cell["fcen"],
        simulator.cell["fwidth"],
        simulator.cell["nfreq"],
    )
    plot_S_parameters(
        S_parameters["S11"],
        S_parameters["S21"],
        lambdas=wavelengths,
        save_path="test_s_parameters_plot.png",
    )

    # Plot E and H fields
    EH_fields = simulator.get_EH_fields()
    dft_wavelengths = 1 / get_freqs(
        simulator.cell["fcen"],
        simulator.cell["fwidth"],
        simulator.cell["dft_nfreqs"],
    )
    plot_EH_fields(EH_fields, dft_wavelengths, save_path="test_eh_fields_plot.png")


if __name__ == "__main__":
    test_sim_1()
