"""Simulator base class and utilities for sweep simulations."""

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple

from auto_all import public


@public
def multiline_join(in_str: str) -> str:
    ix, line = next(
        filter(
            lambda ix_line: len(ix_line[1].lstrip())
            and ix_line[1].lstrip()[0] != ix_line[1][0],
            enumerate(in_str.splitlines()),
        )
    )
    indent_amt = len(line) - len(line.lstrip())
    return "\n".join(
        map(
            lambda e: e[1][
                (0 if e[0] < ix else min(indent_amt, len(e[1]) - len(e[1].lstrip()))) :
            ],
            enumerate(in_str.splitlines()),
        )
    )


@public
@dataclass
class Simulator(ABC):
    """Abstract base class for sweep simulators."""

    config: "SweepConfig"
    netlist_name: str = field(init=False, default="pysweep")
    netlist_ext: str = field(init=False)

    def __post_init__(self):
        sim_name = self.__class__.__name__.lower().replace("simulator", "")
        try:
            subprocess.run(
                ["which", sim_name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            raise EnvironmentError(
                f"{sim_name} not found in PATH. Please install or set up the environment correctly.\n\n{e}"
            )

    def __enter__(self):
        if netlist := self.generate_netlist():
            with open(self.netlist_filepath, "w") as f:
                f.write(netlist)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    @property
    def netlist_filepath(self) -> str:
        return os.path.expandvars(
            f"$TECHSWEEP_DIR{os.sep}{self.netlist_name}{self.netlist_ext}"
        )

    @abstractmethod
    def generate_netlist(self, *args, **kwargs) -> str:
        pass

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict:
        pass

    # @abstractmethod
    # def extract_sweep_params(self, *args, **kwargs) -> Tuple[Dict, Dict]:
    #     """Extracts sweep parameters from the simulation results.
    #     Returns a tuple of dictionaries containing the extracted parameters `(sweep, noise)`.
    #     """
    #     pass

    @abstractmethod
    def cleanup(self, *args, **kwargs) -> None:
        pass
