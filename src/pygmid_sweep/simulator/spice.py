import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from auto_all import public
from PySpice.Probe.WaveForm import WaveForm
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Simulation import CircuitSimulator
from PySpice.Unit import *
from tqdm_loggable.auto import tqdm

from .sim import Simulator

LOGGER = logging.getLogger(__name__)


def setup_worker_logging(q):
    # Each child process needs a QueueHandler to send logs to the main process
    qh = logging.handlers.QueueHandler(q)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(qh)


@public
@dataclass
class SpiceSimulator(Simulator):
    """Spice simulator class for technology sweeps."""

    netlist_ext: str = ".spice"
    circuit: Circuit = field(init=False)

    def __post_init__(self):
        self.circuit = Circuit(self.netlist_name)
        super().__post_init__()

    def __deepcopy__(self, memo: Dict) -> "SpiceSimulator":
        new_sim = self.__class__(config=deepcopy(self.config, memo=memo))
        memo[id(self)] = new_sim
        for k, v in self.__dict__.items():
            if isinstance(v, Circuit):
                setattr(new_sim, k, v.clone())
            else:
                setattr(new_sim, k, deepcopy(v, memo=memo))
        return new_sim

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
                    "{" + f"v{term}" + ("} AC 1" if term == "g" else "}"),
                )

        self.circuit.MOSFET(
            1,
            "d",
            "g",
            self.circuit.gnd,
            "b",
            model=self.config.model,
            width=self.config.width @ u_um,
            m=self.config.nfing,
        )

        self.config.outvar_mapping = self.config.OUTVARS + self.config.OUTVARS_NOISE
        self.circuit.raw_spice += "\n".join(
            map(
                lambda var: f".save {var}",
                ["all", "g", "d", "b", "n", "onoise_total", "inoise_total"]
                + list(map(lambda node: f"@V{node}[dc]", ["g", "d", "b"]))
                + list(self.config.outvar_mapping.keys()),
            )
        )

        return str(self.circuit) if self.config.debug else ""

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

        simulator = self.circuit.simulator()
        return simulator

    def run(self, *args, **kwargs) -> Dict:
        Ls = self.config.length
        VSBs = self.config.vsb

        output_dict = {}
        dimshape = (len(Ls), len(self.config.vgs), len(self.config.vds), len(VSBs))
        for outvar in self.config.OUTVARS:
            output_dict[outvar] = np.zeros(dimshape, order="F")

        for outvar in self.config.OUTVARS_NOISE:
            output_dict[outvar] = np.zeros(dimshape, order="F")

        def map_outputs(sweep_dict, noise_dict, i, j):
            self.config.outvar_mapping = self.config.OUTVARS
            for sim_param, values in sweep_dict.items():
                for outvar, m in self.config.outvar_mapping.get(sim_param, {}).items():
                    output_dict[outvar][i, :, :, j] += np.squeeze(values * m)

            self.config.outvar_mapping = self.config.OUTVARS_NOISE
            for sim_param, values in noise_dict.items():
                for outvar, m in self.config.outvar_mapping.get(sim_param, {}).items():
                    output_dict[outvar][i, :, :, j] += np.squeeze(values * m)

        # Create the multiprocessing spawn context
        mp_ctx = mp.get_context("spawn")

        # Create a multiprocessing queue for logs
        log_queue = mp_ctx.Queue()

        # Create a listener in the main process that reads from the queue
        # and handles logs using the main process's configuration.
        listener = logging.handlers.QueueListener(log_queue, respect_handler_level=True)
        listener.start()

        try:
            with ProcessPoolExecutor(
                mp_context=mp_ctx,
                initializer=setup_worker_logging,
                initargs=(log_queue,),
            ) as executor:
                for i, L in enumerate(tqdm(Ls, desc="Sweeping L")):
                    futures = {
                        executor.submit(deepcopy(self).run_sim, length=L, vsb=VSB): j
                        for j, VSB in enumerate(VSBs)
                    }
                    for future in tqdm(
                        as_completed(futures),
                        desc="Sweeping VSB",
                        total=len(futures),
                        leave=False,
                    ):
                        try:
                            sweep_dict, noise_dict = future.result()
                            j = futures[future]
                        except Exception as e:
                            LOGGER.error(f"Failed to run simulation: {e}")
                            raise e

                        map_outputs(sweep_dict, noise_dict, i, j)
        finally:
            listener.stop()

        # for i, L in enumerate(tqdm(Ls, desc="Sweeping L")):
        #     for j, VSB in enumerate(tqdm(VSBs, desc="Sweeping VSB", leave=False)):
        #         sweep_dict, noise_dict = self.run_sim(length=L, vsb=VSB)
        #         map_outputs(sweep_dict, noise_dict, i, j)

        return output_dict

    def run_sim(self, length, vsb, *args, **kwargs) -> Any:
        shape = (len(self.config.vgs), len(self.config.vds))

        def extract_plot(sim: CircuitSimulator) -> Dict[str, np.typing.NDArray]:
            waveforms = {}
            last_plot = sim.ngspice.plot(sim, sim.ngspice.last_plot)
            for k in self.config.outvar_mapping.keys():
                v = last_plot[k]._data
                if v.shape != (1,):
                    v = np.reshape(v, shape)
                waveforms[k] = v

            assert len(waveforms), "No waveforms found!"
            return waveforms

        def extract_waveform(wv: WaveForm) -> Dict[str, np.typing.NDArray]:
            waveforms = {}
            for k in ["branches", "elements", "internal_parameters", "nodes"]:
                waveforms.update(
                    map(
                        lambda it: (
                            it[0],
                            (
                                np.reshape(np.array(it[1]), shape)
                                if it[1].shape != (1,)
                                else np.array(it[1])
                            ),
                        ),
                        getattr(wv, k).items(),
                    )
                )

            assert len(waveforms), "No waveforms found!"
            return waveforms

        self.config.outvar_mapping = self.config.OUTVARS
        self.circuit["XM1"].length = length @ u_um
        self.circuit.parameter("vg", "0")
        self.circuit.parameter("vd", "0")
        self.circuit.parameter("vb", f"{-vsb:.3f}")

        simulator = self.simulator
        simulator.dc(
            Vd=slice(
                0,
                u_V(self.config.vds_max),
                u_V(self.config.vds_step),
            ),
            Vg=slice(
                0,
                u_V(self.config.vgs_max),
                u_V(self.config.vgs_step),
            ),
        )

        sweep_dict = extract_plot(simulator)

        self.config.outvar_mapping = self.config.OUTVARS_NOISE
        noise_dict: Dict[str, np.typing.NDArray] = {
            key: np.zeros(shape, order="F") for key in self.config.outvar_mapping
        }

        for i, vgs in enumerate(self.config.vgs):
            self.circuit.parameter("vg", f"{vgs:.3f}")
            for j, vds in enumerate(self.config.vds):
                self.circuit.parameter("vd", f"{vds:.3f}")
                simulator = self.simulator
                simulator.noise(
                    "n",
                    self.circuit.gnd,
                    src="Vg",
                    variation="dec",
                    points=10,
                    start_frequency=1 @ u_Hz,
                    stop_frequency=1e11 @ u_Hz,
                    points_per_summary=1,
                )

                for key, value in extract_plot(simulator).items():
                    noise_dict[key][i, j] = value.squeeze()

                # for key, value in filter(
                #     lambda it: it[0] in noise_dict,
                #     extract_waveform(
                #         simulator.noise(
                #             "n",
                #             self.circuit.gnd,
                #             src="Vg",
                #             variation="dec",
                #             points=10,
                #             start_frequency=1 @ u_Hz,
                #             stop_frequency=1e11 @ u_Hz,
                #             points_per_summary=1,
                #         )
                #     ).items(),
                # ):
                #     noise_dict[key][i, j] = value.squeeze()

        return sweep_dict, noise_dict

    def cleanup(self, *args, **kwargs) -> None:
        return


class NGSpiceSimulator(SpiceSimulator):
    # pass
    @property
    def simulator(self):
        simulator = super().simulator
        simulator.ngspice.exec_command("set sqrnoise")
        return simulator


class XyceSimulator(SpiceSimulator):
    def __post_init__(self):
        raise NotImplementedError("Xyce simulator is not implemented yet")
