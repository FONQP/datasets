import meep as mp
from meep.materials import Si, SiO2

_MATERIAL_MAP = {
    "air": mp.air,
    "si": Si,
    "sio2": SiO2,
}


def str_to_material(material_str: str) -> mp.Medium:
    material_str = material_str.lower()
    return _MATERIAL_MAP[material_str]


# The following materials are generated using src/dataset/eps_fit_lorentzian.py

# Si_400_1000
# D. Franta, A. Dubroka, C. Wang, A. Giglia, J. Vohánka, P. Franta, I. Ohlídal. Temperature-dependent dispersion model of float zone crystalline silicon, <a href="https://doi.org/j.apsusc.2017.02.021"><i>Appl. Surf. Sci.</i> <b>421</b>, 405-419 (2017)</a> (Numerical data kindly provided by Daniel Franta)
E_susceptibilities = [
    mp.LorentzianSusceptibility(frequency=3.54528, gamma=0.00000, sigma=8.20540),
    mp.LorentzianSusceptibility(frequency=2.70663, gamma=0.00659, sigma=1.74795),
    mp.LorentzianSusceptibility(frequency=2.61000, gamma=0.39405, sigma=0.65202),
]
Si_400_1000 = mp.Medium(
    epsilon=1.1,
    E_susceptibilities=E_susceptibilities,  # type: ignore
    valid_freq_range=mp.FreqRange(min=1.00000, max=2.50000),
)

_MATERIAL_MAP["si_400_1000"] = Si_400_1000
