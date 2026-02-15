# Datasets

This repository contains dataset links and code for generating simulated datasets for PhotoMizer.

## Installation

```bash
git clone https://github.com/FONQP/datasets.git --depth 1
cd datasets

uv sync
```

## Generating Datasets
A CLI is provided to start dataset simulation as follows:

```bash
uv run gen_dataset -h
```
#### The config formats can be found in [configs](./configs) directory.

#### Check the [slurm](./slurm) directory for slurm scripts to run the dataset generation on a cluster.

For plotting utilities, you can run:

```bash
uv run plot -h
```

## Contributing
Contributions to this repository are welcome! Please follow the standard GitHub flow for contributing.

The tests are in the [tests](./tests) directory. Run the following command to test your changes:

```bash
uv run pytest
```

---

plt.figure(figsize=(15, 5))
for i, component in enumerate(["Ex", "Ey", "Ez"]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(
        np.abs(e[component][0].T),
        origin="lower",
        extent=(
            -cell.x / 2,
            cell.x / 2,
            -cell.y / 2,
            cell.y / 2,
        ),
    )
    plt.colorbar(label=f"|{component}|")
    plt.title(f"{component} Field Magnitude")
    plt.xlabel("x ($u m$)")
    plt.ylabel("y ($u m$)")