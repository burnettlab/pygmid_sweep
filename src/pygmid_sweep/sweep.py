import glob
import logging
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, Generator

from auto_all import public

from pygmid_sweep.config import SweepConfig

LOGGER = logging.getLogger(__name__)


def configs(config_loc: os.PathLike) -> Generator[SweepConfig, None, None]:
    config_loc = Path(config_loc)
    if config_loc.is_dir():
        files = [Path(config_loc) / f for f in glob.glob("*.py", root_dir=config_loc)]
        if not files:
            raise ValueError(f"No .py files found in directory: {config_loc}")
    else:
        files = [config_loc]

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    for config_file in files:
        assert config_file.suffix == ".py", "Config file must be a Python file"
        LOGGER.debug(f"Loading config from: {config_file}")

        module = import_module(
            ".".join(config_file.relative_to(os.getcwd()).with_suffix("").parts)
        )

        for cls in filter(
            lambda c: isinstance(c, type)
            and issubclass(c, SweepConfig)
            and c != SweepConfig
            and getattr(c, "__module__", None) == module.__name__,
            [getattr(module, name) for name in dir(module)],
        ):
            LOGGER.debug(f"Found Config subclass: {cls.__name__}")
            cfg = cls()
            if logging.root.isEnabledFor(
                logging.DEBUG
            ) == cfg.debug and not cfg.lut.sweep_exists(cfg.savefile, **dict(cfg)):
                yield cfg
            else:
                LOGGER.debug(f"Skipping {cls.__name__} ({cfg.model})")


def save_sweep(data: Dict, cfg: SweepConfig):
    cfg_dict = dict(cfg)
    for k in cfg_dict.keys():
        assert k not in data, f"Key collision: {k}"
    data.update(cfg_dict)

    cfg.lut._save(cfg.savefile, data)


@public
def run(config_file_path: os.PathLike, skip_sweep: bool = False) -> None:
    for cfg in configs(config_file_path):
        with cfg.simulator(cfg) as sim:
            if skip_sweep:
                LOGGER.info(f"Skipping sweep: {cfg.savefile}")
                continue
            else:
                # sim.run()
                try:
                    LOGGER.info(f"Sweeping {cfg.__class__.__name__} ({cfg.model})")
                    save_sweep(sim.run(), cfg)
                    LOGGER.info(f"Wrote sweep data to {cfg.savefile}")
                except Exception as e:
                    LOGGER.error(f"Failed to write sweep data: {e}")
