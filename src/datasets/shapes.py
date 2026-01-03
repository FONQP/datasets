import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List

import meep as mp
import numpy as np
from scipy.stats import qmc

from datasets.materials import str_to_material


class ShapeType(Enum):
    CUBOID = "cuboid"
    CYLINDER = "cylinder"
    POLYGON = "polygon"
    RING = "ring"
    RANDOM = "random"


@dataclass
class Shape:
    name: ShapeType
    h: float
    l: float
    a: float
    b: float = 0.0
    material: mp.Medium = mp.air

    def __post_init__(self):
        if self.b == 0.0:
            self.b = self.a

    def check_constraints(self) -> bool:
        if self.l > self.a:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["name"] = data["name"].value
        data.pop("material", None)

        return data


@dataclass
class Cuboid(Shape):
    w: float = 0.0

    def check_constraints(self) -> bool:
        check = super().check_constraints()
        if self.w > self.b:
            return False
        return check


@dataclass
class Cylinder(Shape):
    def check_constraints(self) -> bool:
        check = super().check_constraints()
        return check


@dataclass
class Polygon(Shape):
    n: float = 3.0

    def __post_init__(self):
        super().__post_init__()
        self.n = int(self.n)

    def check_constraints(self) -> bool:
        print(self.l / math.sin(math.pi / self.n), self.a)
        if (self.n < 3) or ((self.l / math.sin(math.pi / self.n)) > self.a):
            return False
        return True


@dataclass
class Ring(Shape):
    i: float = 0.0

    def check_constraints(self) -> bool:
        check = super().check_constraints()
        if self.i > self.l:
            return False
        return check


@dataclass
class RandomShape(Shape):
    cov: float = 0.1

    def check_constraints(self) -> bool:
        return True


class ShapeGenerator:
    def __init__(self, shape_config: Dict):
        self.shape_config = shape_config
        self.sampler_instance = qmc.LatinHypercube(
            d=len(self.shape_config["param_ranges"])
        )

        self.l_bounds = [
            self.shape_config["param_ranges"][p][0] for p in self.shape_config["params"]
        ]
        self.u_bounds = [
            self.shape_config["param_ranges"][p][1] for p in self.shape_config["params"]
        ]

    def get_shape(self, randomize: bool = True) -> Shape:
        MAX_TRIES = 100
        for _ in range(MAX_TRIES):
            if randomize:
                params = self.sampler_instance.random()
                params = qmc.scale(params, self.l_bounds, self.u_bounds)
                params = np.round(params, 3)
                params = {
                    p: params[0][j] for j, p in enumerate(self.shape_config["params"])
                }
            else:
                params = {
                    p: self.l_bounds[j]
                    for j, p in enumerate(self.shape_config["params"])
                }
            params["material"] = str_to_material(self.shape_config["material"])
            shape = get_shape(ShapeType(self.shape_config["name"]), params)

            if shape.check_constraints():
                return shape

        raise ValueError("Could not generate a valid shape with given constraints.")


def get_shape(shape_type: ShapeType, params: Dict) -> Shape:
    if shape_type == ShapeType.CUBOID:
        shape = Cuboid(name=shape_type, **params)
    elif shape_type == ShapeType.CYLINDER:
        shape = Cylinder(name=shape_type, **params)
    elif shape_type == ShapeType.POLYGON:
        shape = Polygon(name=shape_type, **params)
    elif shape_type == ShapeType.RING:
        shape = Ring(name=shape_type, **params)
    elif shape_type == ShapeType.RANDOM:
        shape = RandomShape(name=shape_type, **params)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    return shape


def get_geometric_object(shape: Shape) -> List[mp.GeometricObject]:
    geom: List[mp.GeometricObject]

    if isinstance(shape, Cuboid):
        geom = [
            mp.Block(
                size=mp.Vector3(shape.l, shape.w, shape.h),
                material=shape.material,
            )
        ]

    elif isinstance(shape, Cylinder):
        geom = [
            mp.Cylinder(
                radius=shape.l / 2,
                height=shape.h,
                material=shape.material,
            )
        ]

    elif isinstance(shape, Polygon):
        radius = shape.l / (2 * math.sin(math.pi / shape.n))
        side_vectors = []
        for i in range(int(shape.n)):
            angle = 2 * math.pi * i / shape.n
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            side_vectors.append(mp.Vector3(x, y))

        geom = [
            mp.Prism(
                vertices=side_vectors,
                height=float(shape.h),
                material=shape.material,
            )
        ]

    elif isinstance(shape, Ring):
        outer = mp.Cylinder(
            radius=shape.l / 2,
            height=shape.h,
            material=shape.material,
        )
        inner = mp.Cylinder(
            radius=shape.i / 2,
            height=shape.h,
            material=mp.air,
        )
        geom = [outer, inner]

    elif isinstance(shape, RandomShape):
        N_grains = int(shape.a / shape.l)
        total_grains = N_grains * N_grains
        num_ones = int(shape.cov * total_grains)
        flat_mask = np.zeros(total_grains, dtype=int)
        flat_mask[:num_ones] = 1
        np.random.shuffle(flat_mask)
        grain_weights = flat_mask.reshape((N_grains, N_grains)).astype(float)

        geom = [
            mp.Block(
                size=mp.Vector3(shape.a, shape.a, shape.h),
                material=mp.MaterialGrid(
                    mp.Vector3(N_grains, N_grains, 0),
                    medium1=mp.air,
                    medium2=shape.material,
                    weights=grain_weights,
                    do_averaging=True,
                ),
            )
        ]

    else:
        raise ValueError(f"Unknown shape type: {type(shape)}")

    return geom
