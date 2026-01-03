import os
from argparse import ArgumentParser, Namespace
from datetime import datetime

import toml

from datasets.simulate import simulate_shape
from datasets.utils import logger


def simulate_dataset() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    parser = ArgumentParser(description="Generate a dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/simulated_datasets/data_{timestamp}",
        help="Output directory for the dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args: Namespace = parser.parse_args()

    logger.setLevel(args.log)

    config = toml.load(args.config)

    shape_list = config.get("shape_list", {})

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for shape_name in shape_list:
            shape_dir = os.path.join(args.output_dir, shape_name)
            os.makedirs(shape_dir, exist_ok=True)
            data_dir = os.path.join(shape_dir, "data")
            os.makedirs(data_dir, exist_ok=True)

    for shape_name in shape_list:
        shape_dir = os.path.join(args.output_dir, shape_name)
        simulate_shape(
            shape_name,
            config,
            shape_dir,
        )

    logger.info(f"Dataset generation completed. Output directory: {args.output_dir}")
