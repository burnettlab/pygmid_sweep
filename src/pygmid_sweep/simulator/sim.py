"""Simulator base class and utilities for sweep simulations."""

import logging
import multiprocessing as mp
import os
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Self, Tuple

import numpy as np
from auto_all import public
from PySpice.Probe.WaveForm import WaveForm
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice.Simulation import CircuitSimulator
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
        sim_name = self.__class__.__name__.lower().replace("simulator", "")
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

    # @abstractmethod
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

        simulator = self.circuit.simulator(
            temperature=(self.config.temp - 273.15) @ u_Degree,
        )
        simulator.options(**dict(self.config.simulator_options))
        return simulator

    def generate_netlist(self, *args, **kwargs) -> str:
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

        self.config.outvar_mapping = self.config.OUTVARS + self.config.OUTVARS_NOISE
        self.circuit.raw_spice += os.linesep.join(
            map(
                lambda var: f".save {var}",
                ["all", "g", "d", "b", "n", "onoise_total", "inoise_total"]
                + list(map(lambda node: f"@V{node}[dc]", ["g", "d", "b"]))
                + list(self.config.outvar_mapping.keys()),
            )
        )

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
    # def cleanup(self, *args, **kwargs) -> None:
    #     pass


# @public
# @dataclass
# class SpectreSimulator(Simulator):
#     def __post_init__(self):
#         raise NotImplementedError("Spectre simulator is not implemented yet")


@public
@dataclass
class NGSpiceSimulator(Simulator):
    netlist_ext: str = ".spice"

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

        simulator = self.circuit.simulator(
            ngspice_id=getattr(self, "_id", 0),
            temperature=(self.config.temp - 273.15) @ u_Degree,
        )
        simulator.options(**dict(self.config.simulator_options))
        simulator.ngspice.exec_command("set sqrnoise")
        return simulator

    def run_sim(self, length, vsb, *args, id=None, **kwargs) -> Any:
        if id is not None:
            self._id = id

        # Get the simulator and simulation executor
        simulator = self.simulator

        shape = (len(self.config.vgs), len(self.config.vds))
        self.circuit["XM1"].length = length @ u_um
        self.circuit.parameter("gv", "0")
        self.circuit.parameter("dv", "0")
        self.circuit.parameter("bv", f"{-vsb:.3f}")

        self.config.outvar_mapping = self.config.OUTVARS + self.config.OUTVARS_NOISE
        sweep_dict: Dict[str, np.typing.NDArray] = {
            key: np.zeros(shape, order="F") for key in self.config.outvar_mapping
        }

        # Run DC sweep
        dc = simulator.dc(
            Vg=slice(
                u_V(self.config.vgs_min),
                u_V(self.config.vgs_max),
                u_V(self.config.vgs_step),
            ),
            Vd=slice(
                u_V(self.config.vds_min),
                u_V(self.config.vds_max),
                u_V(self.config.vds_step),
            ),
        )

        self.config.outvar_mapping = self.config.OUTVARS
        for k in self.config.outvar_mapping.keys():
            sweep_dict[k] = np.reshape(np.asarray(dc[k]), shape, order="F")

        simulator.ngspice.destroy()
        self.config.outvar_mapping = self.config.OUTVARS_NOISE
        if not len(self.config.outvar_mapping):
            return sweep_dict

        # Setup raw spice for noise sweep
        node = "g" if len(self.config.vgs) < len(self.config.vds) else "d"

        if node == "g":
            min_v = self.config.vds_min
            max_v = self.config.vds_max
            step_v = self.config.vds_step
            other = "d"
        else:
            min_v = self.config.vgs_min
            max_v = self.config.vgs_max
            step_v = self.config.vgs_step
            other = "g"

        raw_spice_addon = os.linesep + ".noise v(n) vg lin 1 1 1 1"
        self.circuit.raw_spice += raw_spice_addon
        # self.circuit.raw_spice += os.linesep.join(
        #     [
        #         "",
        #         ".noise v(n) vg lin 1 1 1 1",
        #         ".control",
        #         "set sqrnoise",
        #         "set wr_singlescale",
        #         f"compose v{other}_vec start={min_v:.3f} stop={(max_v + step_v/2):.3f} step={step_v:.3f}",
        #         f"foreach var1 $&v{other}_vec",
        #         f"    alterparam {other}v $var1",
        #         "    run",
        #         "end",
        #         ".endcontrol",
        #         "",
        #     ]
        # )

        # Run "fake" op sims to get noise output
        simulator.operating_point()
        for i, vg in enumerate(
            self.config.vgs
            if id is not None
            else tqdm(self.config.vgs, desc="Sweeping VGS", leave=False)
        ):
            simulator.ngspice.exec_command(f"alterparam gv={vg:.3f}")
            for j, vd in enumerate(
                self.config.vds
                if id is not None
                else tqdm(self.config.vds, desc="Sweeping VDS", leave=False)
            ):
                simulator.ngspice.exec_command(f"alterparam dv={vd:.3f}")
                simulator.ngspice.exec_command("reset")
                simulator.ngspice.exec_command("run")

                for sim_output in map(
                    lambda k: simulator.ngspice.plot(simulator, k),
                    filter(
                        lambda k: k.startswith("noise"), simulator.ngspice.plot_names
                    ),
                ):
                    vg_ix = np.where(
                        np.isclose(sim_output["@vg[dc]"]._data, self.config.vgs)
                    )[0].item()
                    vd_ix = np.where(
                        np.isclose(sim_output["@vd[dc]"]._data, self.config.vds)
                    )[0].item()

                    for k in self.config.outvar_mapping.keys():
                        sweep_dict[k][i, j] = sim_output[k]._data.item()

                simulator.ngspice.destroy()
                simulator.ngspice.exec_command("destroy all")
        # Reset circuit.rawspice
        self.circuit.raw_spice = self.circuit.raw_spice.replace(raw_spice_addon, "")
        simulator.ngspice.reset()
        return sweep_dict

        sweep = (
            self.config.vgs
            if len(self.config.vgs) < len(self.config.vds)
            else self.config.vds
        )
        other_sweep = (
            self.config.vds
            if len(self.config.vgs) < len(self.config.vds)
            else self.config.vgs
        )

        # Run noise sweep
        for value in (
            sweep
            if id is not None
            else tqdm(
                sweep,
                desc=f"Sweeping V{node.upper()}S",
                total=len(sweep),
                leave=False,
            )
        ):
            # self.circuit.parameter("vg", "0")
            simulator.ngspice.exec_command(f"alterparam {node}v={value:.3f}")
            simulator.ngspice.exec_command("reset")
            simulator.ngspice.exec_command("run")
            # simulator.operating_point()

            for sim_output in map(
                lambda k: simulator.ngspice.plot(simulator, k),
                filter(lambda k: k.startswith("noise"), simulator.ngspice.plot_names),
            ):
                vg_ix = np.where(
                    np.isclose(sim_output["@vg[dc]"]._data, self.config.vgs)
                )[0].item()
                vd_ix = np.where(
                    np.isclose(sim_output["@vd[dc]"]._data, self.config.vds)
                )[0].item()

                for k in self.config.outvar_mapping.keys():
                    sweep_dict[k][vg_ix, vd_ix] = sim_output[k]._data.item()

            simulator.ngspice.destroy()
            simulator.ngspice.exec_command("destroy all")

        # Reset circuit.rawspice
        while ".noise" in self.circuit.raw_spice:
            pre, post = self.circuit.raw_spice.split(".noise")
            _, post = post.split(".endcontrol")
            self.circuit.raw_spice = f"{pre}{post}".strip()
        simulator.ngspice.reset()

        return sweep_dict


# @public
# @dataclass
# class XyceSimulator(Simulator):
#     def __post_init__(self):
#         raise NotImplementedError("Xyce simulator is not implemented yet")
