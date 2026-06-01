import logging
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pygmid

from pygmid_sweep.sweep import run as pygmid_sweep_run

try:
    __version__ = version("pygmid")
except PackageNotFoundError:
    __version__ = "unknown"


def cli():
    from argparse import ArgumentParser

    description = "CLI for running Pygmid techsweeps"
    parser = ArgumentParser(description=description)
    parser.add_argument("--version", action="version", version=f"pygmid {__version__}")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to directory or .py file containing sweep configuration",
    )
    parser.add_argument(
        "--logging",
        type=str,
        default=str(Path(__file__).parent / "logging.yaml"),
        help="Logging configuration file path",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running the simulation, e.g. for testing.",
    )

    args = parser.parse_args()

    pygmid.logging.setup_logging(default_path=args.logging)
    if __name__ == "__main__":
        logging.root.setLevel(logging.DEBUG)
    LOGGER = logging.getLogger("pygmid_sweep")

    if args.config is None:
        LOGGER.error(
            "Please provide a config file with --config if using the sweep mode"
        )
        sys.exit(-1)

    config_path = Path(os.path.abspath(os.path.expandvars(args.config)))
    if not config_path.exists():
        LOGGER.error(f"Config file not found: {config_path}")
        sys.exit(-1)

    pygmid_sweep_run(
        str(config_path.resolve()),
        skip_sweep=getattr(args, "skip_run", False),
    )


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    cli()
