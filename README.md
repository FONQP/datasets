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