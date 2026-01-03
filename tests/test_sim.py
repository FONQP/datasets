import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import toml

from datasets.shapes import ShapeGenerator
from datasets.simulator import Simulator


def test_sim_1():
    config = toml.load("test_config.toml")
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

    S_parameters = simulator.get_S_parameters()

    cell = simulator.cell["cell_size"]
    vertical_slice = mp.Volume(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(cell.x, 0, cell.z),
        dims=2,
    )

    simulator.sim.plot2D(
        output_plane=vertical_slice,
        fields=mp.Ex,
        eps_parameters={"cmap": "binary", "interpolation": "none"},
        show_sources=True,
        show_monitors=True,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        S_parameters["wavelengths"],
        np.abs(S_parameters["S11"]) ** 2,
        label="$|S_{11}|^2$ (Reflection)",
    )
    plt.plot(
        S_parameters["wavelengths"],
        np.abs(S_parameters["S21"]) ** 2,
        label="$|S_{21}|^2$ (Transmission)",
    )
    plt.xlabel("Wavelength ($u m$)")
    plt.ylabel("Magnitude")
    plt.title("S-Parameters")
    plt.legend()
    plt.grid(True)
    plt.show()
