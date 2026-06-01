import concurrent.futures
import glob
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import psf_utils
from auto_all import public
from tqdm.auto import tqdm

from .sim import Simulator

LOGGER = logging.getLogger(__name__)


@public
@dataclass
class SpectreSimulator(Simulator):
    """
    Spectre simulator class for technology sweeps.

    *** THIS CLASS NEEDS UPDATING AFTER CHANGES TO CONFIG AND SIMULATOR CLASSES
    """

    netlist_ext: str = ".scs"
    args: List[str] = field(init=False)

    def __post_init__(self):
        self.args = [
            "+escchars",
            "=log",
            f"{self._sweep_dir}/psf/spectre.out",
            "-format",
            "psfascii",
            "-raw",
            f"{self._sweep_dir}/psf",
        ]
        super().__post_init__()

    @property
    def output(self) -> str:
        if getattr(self, "_output", None) is None:
            self._output = f"{self._sweep_dir}/psf_0_0"
        return self._output

    @output.setter
    def output(self, args: Tuple):
        i, j = map(
            lambda a: a[0] if a[0] is not None else a[1],
            zip(args, self.output.split("_")[-2:]),
        )
        self._output = f"{self._sweep_dir}/psf_{i}_{j}"

    def generate_netlist(self, **kwargs) -> str:
        return (
            f"""//{self.netlist_name}.scs
include {kwargs['modelfile']}
include "{kwargs['paramfile']}"

save *:oppoint

parameters gs=0.0 ds=0.0 L=length*1e-6 Wtot={kwargs['width']}e-6 W={kwargs['width']/kwargs['NFING']}e-6 nf={kwargs['NFING']}

vnoi     (vx  0)         vsource dc=0
vdsn     (vdn vx)         vsource dc=ds
vgsn     (vgn 0)         vsource dc=gs
vbsn     (vbn 0)         vsource dc=-sb
vdsp     (vdp vx)         vsource dc=-ds
vgsp     (vgp 0)         vsource dc=-gs
vbsp     (vbp 0)         vsource dc=sb


mp (vdp vgp 0 vbp) {kwargs['modelp']} {kwargs['mp_supplement']}

mn (vdn vgn 0 vbn) {kwargs['modeln']} {kwargs['mn_supplement']}

simOptions options gmin=1e-13 reltol=1e-4 vabstol=1e-6 iabstol=1e-10 temp={kwargs['temp']-273.15} tnom=27
sweepvds sweep param=ds start=0 stop={kwargs['VDS_max']} step={kwargs['VDS_step']} """
            + "{\n"
            + f"\tsweepvgs dc param=gs start=0 stop={kwargs['VGS_max']} step={kwargs['VGS_step']}"
            + "\n}"
            + f"""
sweepvds_noise sweep param=ds start=0 stop={kwargs['VDS_max']} step={kwargs['VDS_step']} """
            + "{\n"
            + f"\tsweepvgs_noise noise freq=1 oprobe=vnoi param=gs start=0 stop={kwargs['VGS_max']} step={kwargs['VGS_step']}"
            + "\n}"
        )

    def run(self):
        Ls = self._config["SWEEP"]["LENGTH"]
        VSBs = self._config["SWEEP"]["VSB"]

        nch = self._config.generate_m_dict()
        pch = self._config.generate_m_dict()
        dimshape = (len(Ls), len(nch["VGS"]), len(nch["VDS"]), len(VSBs))
        for outvar in self._config["outvars"]:
            nch[outvar] = np.zeros(dimshape, order="F")
            pch[outvar] = np.zeros(dimshape, order="F")

        for outvar in self._config["outvars_noise"]:
            nch[outvar] = np.zeros(dimshape, order="F")
            pch[outvar] = np.zeros(dimshape, order="F")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            # A list to store futures for data parsing
            futures = []
            for i, L in enumerate(tqdm(Ls, desc="Sweeping L")):
                for j, VSB in enumerate(tqdm(VSBs, desc="Sweeping VSB", leave=False)):
                    self._config.write_params(length=L, sb=VSB, index=(i, j))
                    self.directory = self.output
                    self._run_sim()

                    futures.append(executor.submit(self.parse_sim, *[self.output]))

            concurrent.futures.wait(futures)

        for f in futures:
            i, j, n_dict, p_dict, nn_dict, pn_dict = f.result()
            for n, p in zip(self._config["n"], self._config["p"]):
                params_n = n
                values_n = n_dict[params_n[0]]
                params_p = p
                values_p = p_dict[params_p[0]]
                for m, outvar in enumerate(self._config["outvars"]):
                    nch[outvar][i, :, :, j] += np.squeeze(values_n * params_n[2][m])
                    pch[outvar][i, :, :, j] += np.squeeze(values_p * params_p[2][m])

            for n, p in zip(self._config["n_noise"], self._config["p_noise"]):
                params_n = n
                values_n = nn_dict[params_n[0]]
                params_p = p
                values_p = pn_dict[params_p[0]]
                for m, outvar in enumerate(self._config["outvars_noise"]):
                    nch[outvar][i, :, :, j] += np.squeeze(values_n * params_n[2])
                    pch[outvar][i, :, :, j] += np.squeeze(values_p * params_p[2])

        return self._cleanup(nch, pch)

    def _run_sim(self):
        try:
            cmd_args = ["spectre", self.netlist_filepath] + [*self.args]
            LOGGER.debug(f"Running command: {' '.join(cmd_args)}")
            subprocess.run(
                cmd_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError:
            LOGGER.exception("Error executing process!")

    def _cleanup(self, nch, pch) -> Tuple[str, ...]:
        try:
            os.remove("params.scs")
        except OSError:
            LOGGER.exception("Could not perform cleanup!")
        return super()._cleanup(nch, pch)

    def parse_sim(self, filepath):
        fileparts = filepath.split("_")
        i = int(fileparts[-2])
        j = int(fileparts[-1])

        n_dict, p_dict, nn_dict, pn_dict = self.extract_sweep_params(filepath)

        return i, j, n_dict, p_dict, nn_dict, pn_dict

    def extract_sweep_params(self, sweep_output_directory):
        def extract_number_regex(string):
            pattern = r"\d+"  # Matches one or more digits
            match = re.search(pattern, string)
            if match:
                return int(match.group())  # Extracted number as an integer
            return float("inf")

        outputs = []
        for sweep_type in ["DC", "NOISE"]:
            if sweep_type == "DC":
                filename_pattern = "sweepvds-*_sweepvgs.dc"
                params = [".".join(k[0].split(".")[1:]) for k in self._config["n"]]
            elif sweep_type == "NOISE":
                filename_pattern = "sweepvds_noise-*_sweepvgs_noise.noise"
                params = [
                    ".".join(k[0].split(".")[1:]) for k in self._config["n_noise"]
                ]
            else:
                raise ValueError(
                    f"Unknown sweep type: {sweep_type}. Must be 'DC' or 'NOISE'."
                )

            file_paths = glob.glob(
                os.path.join(sweep_output_directory, filename_pattern)
            )
            # remove directory in case it contains number. Only want to sort based on filename itself
            filelist = sorted(
                (os.path.basename(f) for f in file_paths), key=extract_number_regex
            )

            nmos = {
                f"mn.{param}": np.zeros(
                    (
                        len(self._config["SWEEP"]["VGS"]),
                        len(self._config["SWEEP"]["VDS"]),
                    )
                )
                for param in params
            }
            pmos = {
                f"mp.{param}": np.zeros(
                    (
                        len(self._config["SWEEP"]["VGS"]),
                        len(self._config["SWEEP"]["VDS"]),
                    )
                )
                for param in params
            }
            for VDS_i, f in enumerate(filelist):
                # reconstruct path
                file_path = os.path.join(sweep_output_directory, f)
                # need to extract parameter from PSFs
                psf = psf_utils.PSF(file_path)

                for param in params:
                    nmos[f"mn.{param}"][:, VDS_i] = (
                        psf.get_signal(f"mn.{param}").ordinate
                    ).T
                    pmos[f"mp.{param}"][:, VDS_i] = (
                        psf.get_signal(f"mp.{param}").ordinate
                    ).T

            outputs.extend([nmos, pmos])

        return outputs
