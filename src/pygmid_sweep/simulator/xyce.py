"""Xyce simulator class for technology sweeps."""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from auto_all import public
from PySpice.Unit import *
from tqdm_loggable.auto import tqdm

from .sim import Simulator

LOGGER = logging.getLogger(__name__)


@public
@dataclass
class XyceSimulator(Simulator):
    simulator_command: str = "Xyce"
    netlist_ext: str = ".spice"

    def get_simulator_options(self) -> dict[str, Any]:
        return dict(
            temperature=(self.config.temp - 273.15) @ u_Degree,
            simulator="xyce-parallel",
        )

    def run_sim(self, length, vsb, *args, id=None, **kwargs) -> Any:
        if id is not None:
            self._id = id

        # Get the simulator and simulation executor
        simulator = self.simulator

        shape = (len(self.config.vgs), len(self.config.vds))
        self.circuit["M1"].length = length @ u_um
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

        # simulator.ngspice.destroy()
        self.config.outvar_mapping = self.config.OUTVARS_NOISE
        if not len(self.config.outvar_mapping):
            return sweep_dict

        # Setup raw spice for noise sweep
        node = "g" if len(self.config.vgs) < len(self.config.vds) else "d"
        if node == "g":
            min_v = self.config.vds_min
            max_v = self.config.vds_max
            step_v = self.config.vds_step
            sweep = self.config.vgs
            other_sweep = self.config.vds
            other = "d"
        else:
            min_v = self.config.vgs_min
            max_v = self.config.vgs_max
            step_v = self.config.vgs_step
            sweep = self.config.vds
            other_sweep = self.config.vgs
            other = "g"

        self.circuit.raw_spice += os.linesep.join(
            [
                "",
                ".noise v(n) vg lin 1 1 1 1",
                ".control",
                "set sqrnoise",
                "set wr_singlescale",
                f"compose v{other}_vec start={min_v:.3f} stop={(max_v + step_v/2):.3f} step={step_v:.3f}",
                f"foreach var1 $&v{other}_vec",
                f"    alterparam {other}v $var1",
                "    run",
                "end",
                ".endcontrol",
                "",
            ]
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
            self.circuit.parameter(f"v{node}", f"{value:.3f}")
            _ = simulator.operating_point()
            # simulator.ngspice.exec_command(f"alterparam {node}v={value:.3f}")
            # simulator.ngspice.exec_command("reset")
            # simulator.ngspice.exec_command("run")

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
