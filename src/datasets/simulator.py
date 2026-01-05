from typing import Any, Dict

import meep as mp
import numpy as np

from datasets.materials import str_to_material
from datasets.shapes import Shape, get_geometric_object
from datasets.utils import logger


def make_cell(shape: Shape, config: Dict) -> Dict[str, Any]:
    # Simulation domain
    lambda_start = config["simulation"]["lambda_start"]
    lambda_end = config["simulation"]["lambda_end"]

    lambda_width = lambda_end - lambda_start
    nfreq = int(lambda_width // config["simulation"]["lambda_step"]) + 1

    f_min = 1.0 / lambda_end
    f_max = 1.0 / lambda_start

    f_center = 0.5 * (f_min + f_max)
    f_width = f_max - f_min

    # Simulation cell
    cell_height = (
        config["cell"]["airgap"]
        + config["substrate"]["h"]
        + 2 * config["cell"]["pml"]["h"]
    )
    base_z = -cell_height / 2
    xy_plane = mp.Vector3(shape.a, shape.b, 0)

    sim_cell = xy_plane + mp.Vector3(0, 0, cell_height)

    # Substrate
    subs = mp.Block(
        center=mp.Vector3(
            0,
            0,
            base_z
            + ((config["cell"]["pml"]["h"] + config["substrate"]["h"] - 1.0) / 2),
        ),
        size=mp.Vector3(
            mp.inf, mp.inf, config["cell"]["pml"]["h"] + config["substrate"]["h"] + 1.0
        ),
        material=str_to_material(config["substrate"]["material"]),
    )

    # Meta atom
    shape_center = mp.Vector3(
        0,
        0,
        base_z + config["cell"]["pml"]["h"] + config["substrate"]["h"] + shape.h / 2,
    )

    geom = get_geometric_object(shape)
    for obj in geom:
        obj.center = shape_center

    # Sources
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(frequency=f_center, fwidth=f_width),
            center=mp.Vector3(
                0,
                0,
                base_z
                + config["cell"]["pml"]["h"]
                + config["substrate"]["h"]
                + (shape.h / 3)
                + (4 * config["cell"]["airgap"] / 6),
            ),
            size=xy_plane,
            direction=mp.Z,
            eig_kpoint=mp.Vector3(0, 0, -1),
            eig_band=1,
        )
    ]

    # Flux regions
    flux_regions = {}
    flux_regions["s11"] = mp.ModeRegion(
        center=mp.Vector3(
            0,
            0,
            base_z
            + config["cell"]["pml"]["h"]
            + config["substrate"]["h"]
            + (shape.h / 6)
            + (5 * config["cell"]["airgap"] / 6),
        ),
        size=xy_plane,
    )
    flux_regions["s21"] = mp.ModeRegion(
        center=mp.Vector3(
            0, 0, base_z + config["cell"]["pml"]["h"] + (config["substrate"]["h"] / 2)
        ),
        size=xy_plane,
    )

    cell: Dict[str, Any] = {
        "fcen": f_center,
        "fwidth": f_width,
        "nfreq": nfreq,
        "pml_layers": [mp.PML(thickness=config["cell"]["pml"]["h"], direction=mp.Z)],
        "airgap": config["cell"]["airgap"],
        "geometry": [subs] + geom,
        "sources": sources,
        "flux_regions": flux_regions,
        "cell_size": sim_cell,
        "stopping_ref": mp.Vector3(
            0, 0, base_z + config["cell"]["pml"]["h"] + (config["substrate"]["h"] / 6)
        ),
    }

    return cell


class Simulator:
    def __init__(self, shape: Shape, config: Dict) -> None:
        self.cell = make_cell(shape=shape, config=config)
        self.default_material = config.get("default_material", mp.air)
        self.resolution = config["simulation"].get("resolution", 50)

        logger.debug(
            f"Simulator initialized with shape: {shape.name}, and geometry: {self.cell['geometry']}"
        )

    def get_sim_cell_generics(self) -> Dict[str, Any]:
        sources = {
            "source_name": self.cell["sources"][0].src.__class__.__name__,
            "frequency": self.cell["sources"][0].src.frequency,
            "fwidth": 1 / self.cell["sources"][0].src.width,
            "center": (
                self.cell["sources"][0].center.x,
                self.cell["sources"][0].center.y,
                self.cell["sources"][0].center.z,
            ),
        }

        return {
            "fcen": self.cell["fcen"],
            "fwidth": self.cell["fwidth"],
            "nfreq": self.cell["nfreq"],
            "pml_layers": self.cell["pml_layers"],
            "airgap": self.cell["airgap"],
            "cell_size": self.cell["cell_size"],
            "sources": sources,
        }

    def init_simulation_instance(self, empty: bool = False) -> mp.Simulation:
        self.sim = mp.Simulation(
            cell_size=self.cell["cell_size"],
            boundary_layers=self.cell["pml_layers"],
            sources=self.cell["sources"],
            geometry=[] if empty else self.cell["geometry"],
            default_material=self.default_material,
            resolution=self.resolution,
            symmetries=self.cell.get("symmetries", []),
            k_point=mp.Vector3(),
            Courant=0.3,
        )

        self.s11_monitor = self.sim.add_mode_monitor(
            self.cell["fcen"],
            self.cell["fwidth"],
            self.cell["nfreq"],
            self.cell["flux_regions"]["s11"],
        )
        self.s21_monitor = self.sim.add_mode_monitor(
            self.cell["fcen"],
            self.cell["fwidth"],
            self.cell["nfreq"],
            self.cell["flux_regions"]["s21"],
        )

        return self.sim

    def run_empty(self) -> None:
        logger.debug("Running empty simulation for normalization...")
        self.sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                20, mp.Ex, self.cell["stopping_ref"], 1e-9
            ),
        )

        res_norm = self.sim.get_eigenmode_coefficients(self.s21_monitor, [1])
        self.incident = np.array([coef[1] for coef in res_norm.alpha[0]])
        del self.sim

    def run(self, dt=20, decay_by=1e-7) -> None:
        logger.debug("Running full simulation for normalization...")
        self.sim.run(
            until_after_sources=mp.stop_when_fields_decayed(
                dt, mp.Ex, self.cell["stopping_ref"], decay_by
            ),
        )

    def get_S_parameters(self) -> Dict[str, np.ndarray]:
        logger.debug("Getting S parameters...")
        res_s11 = self.sim.get_eigenmode_coefficients(self.s11_monitor, [1])
        reflected = np.array([coef[0] for coef in res_s11.alpha[0]])

        res_s21 = self.sim.get_eigenmode_coefficients(self.s21_monitor, [1])
        transmitted = np.array([coef[1] for coef in res_s21.alpha[0]])

        frequencies = np.linspace(
            self.cell["fcen"] - self.cell["fwidth"] / 2,
            self.cell["fcen"] + self.cell["fwidth"] / 2,
            self.cell["nfreq"],
        )
        lambdas = 1.0 / frequencies

        mask = np.abs(self.incident) > 1e-10
        S11 = np.zeros_like(self.incident)
        S21 = np.zeros_like(self.incident)
        S11[mask] = reflected[mask] / self.incident[mask]
        S21[mask] = transmitted[mask] / self.incident[mask]

        return {"wavelengths": lambdas, "S11": S11, "S21": S21}
