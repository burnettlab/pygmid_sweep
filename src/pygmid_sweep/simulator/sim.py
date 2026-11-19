"""Simulator base class and utilities for sweep simulations."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Self

import numpy as np
from auto_all import public
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from tqdm_loggable.auto import tqdm

LOGGER = logging.getLogger(__name__)


def setup_worker_logging(q):
    # Each child process needs a QueueHandler to send logs to the main process
    qh = logging.handlers.QueueHandler(q)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(qh)


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
    circuit: Circuit = field(init=False)

    def __post_init__(self):
        self.circuit = Circuit(self.netlist_name)
        self.generate_circuit()
        sim_name = getattr(self, "simulator_command")
        try:
            subprocess.run(
                ["which", sim_name],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            LOGGER.error(
                f"{sim_name} not found in PATH. Please install or set up the environment correctly.\n\n{e}"
            )
            raise EnvironmentError(
                f"{sim_name} not found in PATH. Please install or set up the environment correctly.\n\n{e}"
            )

    def __deepcopy__(self, memo: Dict) -> Self:
        new_sim = self.__class__(config=deepcopy(self.config, memo=memo))
        memo[id(self)] = new_sim
        for k, v in self.__dict__.items():
            if isinstance(v, Circuit):
                setattr(new_sim, k, v.clone())
            else:
                setattr(new_sim, k, deepcopy(v, memo=memo))
        return new_sim

    def __enter__(self):
        if netlist := self.generate_netlist():
            with open(self.netlist_filepath, "w") as f:
                f.write(netlist)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    @property
    def netlist_filepath(self) -> str:
        return os.path.expandvars(
            f"$TECHSWEEP_DIR{os.sep}{self.netlist_name}{self.netlist_ext}"
        )

    @abstractmethod
    def get_simulator_options(self) -> dict[str, Any]:
        pass

    @property
    def simulator(self):
        for inc in filter(
            lambda inc: os.path.expandvars(inc) not in self.circuit._includes,
            self.config.include,
        ):
            self.circuit.include(os.path.expandvars(inc))

        for lib, section in filter(
            lambda l: (os.path.expandvars(l[0]), l[1]) not in self.circuit._libs,
            self.config.lib,
        ):
            self.circuit.lib(os.path.expandvars(lib), section=section)

        simulator = self.circuit.simulator(**self.get_simulator_options())
        simulator.options(**dict(self.config.simulator_options))

        # Set up saved values
        self.config.outvar_mapping = self.config.OUTVARS + self.config.OUTVARS_NOISE
        simulator.save_internal_parameters(*self.config.outvar_mapping.keys())

        return simulator

    def generate_circuit(self):
        for inc in self.config.include:
            self.circuit.include(os.path.expandvars(inc))

        for lib, section in self.config.lib:
            self.circuit.lib(os.path.expandvars(lib), section=section)

        if self.config.device == "n":
            plus_node = lambda term: term
            minus_node = lambda _: self.circuit.gnd
        else:
            plus_node = lambda _: self.circuit.gnd
            minus_node = lambda term: term

        for term in ["g", "d", "b", "n"]:

            if term == "n":
                self.circuit.H(term, plus_node(term), minus_node(term), "Vd", 1)
            else:
                self.circuit.V(
                    term,
                    plus_node(term),
                    minus_node(term),
                    "{" + f"{term}v" + ("} AC 1" if term == "g" else "}"),
                )
                self.circuit.parameter(f"v{term}", 0)

        self.circuit.MOSFET(
            1,
            "d",
            "g",
            self.circuit.gnd,
            "b",
            model=self.config.model,
            width=self.config.width @ u_um,
            m=self.config.nfing,
            raw_spice=self.config.model_supplement,
        )

    def generate_netlist(self, *args, **kwargs) -> str:
        return str(self.circuit) if self.config.debug else ""

    def run(self, *args, **kwargs) -> Dict:
        def map_outputs(sweep_dict, i, j):
            self.config.outvar_mapping = self.config.OUTVARS + self.config.OUTVARS_NOISE
            for sim_param, values in sweep_dict.items():
                for outvar, m in self.config.outvar_mapping.get(sim_param, {}).items():
                    output_dict[outvar][i, :, :, j] += np.squeeze(values * m)

        Ls = self.config.length
        VSBs = self.config.vsb

        output_dict = {}
        dimshape = (len(Ls), len(self.config.vgs), len(self.config.vds), len(VSBs))
        for outvar in self.config.OUTVARS:
            output_dict[outvar] = np.zeros(dimshape, order="F")

        for outvar in self.config.OUTVARS_NOISE:
            output_dict[outvar] = np.zeros(dimshape, order="F")

        if self.config.sweep_parallel:
            raise NotImplementedError("Parallel sweep is not implemented")
            # Create the multiprocessing spawn context
            mp_ctx = mp.get_context("spawn")

            # Create a multiprocessing queue for logs
            log_queue = mp_ctx.Queue()

            # Create a listener in the main process that reads from the queue
            # and handles logs using the main process's configuration.
            listener = logging.handlers.QueueListener(
                log_queue, respect_handler_level=True
            )
            listener.start()

            try:
                with ProcessPoolExecutor(
                    mp_context=mp_ctx,
                    initializer=setup_worker_logging,
                    initargs=(log_queue,),
                ) as executor:
                    for i, L in enumerate(tqdm(Ls, desc="Sweeping L")):
                        futures = {
                            executor.submit(
                                deepcopy(self).run_sim, length=L, vsb=VSB, id=j
                            ): j
                            for j, VSB in enumerate(VSBs)
                        }
                        for future in tqdm(
                            as_completed(futures),
                            desc="Sweeping VSB",
                            total=len(futures),
                            leave=False,
                        ):
                            try:
                                sweep_dict = future.result()
                                j = futures[future]
                            except Exception as e:
                                LOGGER.error(f"Failed to run simulation: {e}")
                                raise e

                            map_outputs(sweep_dict, i, j)
            finally:
                listener.stop()
        else:
            for i, L in enumerate(tqdm(Ls, desc="Sweeping L")):
                for j, VSB in enumerate(tqdm(VSBs, desc="Sweeping VSB", leave=False)):
                    map_outputs(self.run_sim(length=L, vsb=VSB), i, j)

        return output_dict

    @abstractmethod
    def run_sim(self, length, vsb, *args, id=None, **kwargs) -> Any:
        pass

    # @abstractmethod
    def cleanup(self, *args, **kwargs) -> None:
        pass
