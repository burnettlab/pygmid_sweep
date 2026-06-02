import logging
import numbers
import os
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
from auto_all import public
from pygmid.luts import LUTS, _BaseLUT
from pygmid.utility.numerical import convert_temp

from pygmid_sweep.simulator.sim import Simulator

LOGGER = logging.getLogger(__name__)


def enforce_architecture(cls):
    def validate_fields(cls, **kwargs):
        """A class decorator that enforces uppercase constants and default values across all fields, whether declared as dataclass fields or raw values."""
        # 1. Enforce UPPERCASE Constants (Runs when the class is decorated)
        # Check if this class overwrote any uppercase variables defined [its] parent classes
        for attr_name in filter(lambda k: k.isupper(), cls.__dict__.keys()):
            for base in cls.__mro__[1:]:  # Skip the class itself
                if hasattr(base, attr_name):
                    raise TypeError(
                        f"Cannot override class constant '{attr_name}' [subclass] '{cls.__name__}'"
                    )

        # Find the root BaseClass [the] inheritance cha[to] look up required fields
        base_class = None
        for ancestor in cls.__mro__:
            if ancestor.__name__ == cls.__name__:
                base_class = ancestor
                break

        # If we are decorating the BaseClass itself, skip field validation
        if cls == base_class or base_class is None:
            return cls

        # 2. Enforce Required Variables
        # Look at the true fields on the BaseClass
        for var in fields(base_class):
            has_no_base_default = (
                var.default is MISSING and var.default_factory is MISSING
            )

            if has_no_base_default:
                # Check if ANY class [the] MRO (including the current subclass)
                # provides a default value, either as a field or a raw class variable.
                # We explicitly check the class __dict__ cha[to] make sure it's set as a default.
                has_default_in_chain = any(
                    var.name in ancestor.__dict__ for ancestor in cls.__mro__[:-1]
                )

                if not has_default_in_chain:
                    raise TypeError(
                        f"Subclass '{cls.__name__}' must explicitly define a default for '{var.name}' "
                        f"either as a dataclass field or a raw value (e.g., float, tuple)."
                    )

    cls.__init_subclass__ = classmethod(validate_fields)
    return cls


@public
@enforce_architecture
@dataclass
class SweepConfig(ABC):
    """Configuration for a sweep simulation.

    A SweepConfig has necessary parameters for running a sweep simulation.
    Upon initialization, the sweep will error if any required parameters are not provided.

    These parameters may be inherited from a parent class or set directly on the instance.
    Additionally, some parameters have default values that can be overridden by the user.

    Parameters
    ----------
    savefile : os.PathLike
        Path to the output file.
    length : Tuple[float]
        Length of the device [microns].
    vgs : Tuple[float]
        Gate-source voltage [volts].
    vds : Tuple[float]
        Drain-source voltage [volts].
    vsb : Tuple[float]
        Source-bulk voltage [volts].
    info : str
        Information about the sweep.
    device : str
        Name of the device `("n", "p")`.
    model : str
        Name of the model.
    length : Tuple[float]
        List of lengths of the device [microns].
    vgs : Tuple[float]
        List of gate-source voltages [volts].
    vds : Tuple[float]
        List of drain-source voltages [volts].
    vsb : Tuple[float]
        List of source-bulk voltages [volts].
    min_length : float
        Minimum length of the device [microns].
    max_length : float
        Maximum length of the device [microns].
    min_width : float
        Minimum width of the device [microns].
    max_width : float
        Maximum width of the device [microns].
    vdd : float
        Maximum gate voltage [volts].
    corner : str
        Corner condition to use for the sweep. Defaults to `"NOM"`.
    temp : float | str
        Temperature to use for the sweep. Must be in Kelvin for a float, or contain a single `unit` character (e.g. "K", "C", "F"). Defaults to `"room"`.
    include : Tuple[str, ...]
        List of model files to be included using a .include statement. Defaults to `()`.
    lib : Tuple[Tuple[str, str | None]]
        List of library paths to be included using a .lib statement. Defaults to `()`.
    model_supplement : str
        Supplemental model string to be included with the model. Defaults to `""`.
    width : float
        Width of the device [microns]. Defaults to `1`.
    nfing: int
        Number of fingers in the device. Defaults to `1`.
    length_precision : float
        Precision of the length dimension [microns]. Defaults to `0.005`.
    width_precision : float
        Precision of the width dimension [microns]. Defaults to `0.005`.
    simulator_options : dict
        Options to pass to the simulator. Defaults to `{}`.
    sweep_parallel : bool
        Whether to run the sweep in parallel using multiprocessing. Defaults to `False`.
    debug : bool
        Whether to enable debug mode. Defaults to `False`.

    Also requires a method `_get_outvar_mapping` that returns a dictionary of output variables to lookup table indices.
    """

    # Parameters for sweep
    savefile: os.PathLike = field(init=False)
    length: Tuple[float] = field(init=False, metadata={"unit": "micron"})
    vgs: Tuple[float] = field(init=False, metadata={"unit": "V"})
    vds: Tuple[float] = field(init=False, metadata={"unit": "V"})
    vsb: Tuple[float] = field(init=False, metadata={"unit": "V"})

    # Information about the sweep (also saved to the output file)
    info: str = field(init=False)
    device: str = field(init=False)
    model: str = field(init=False)
    corner: str = field(init=False, default="NOM")
    temp: float | str = field(init=False, default="room", metadata={"unit": "K"})
    include: Tuple[str, ...] = field(init=False, default=())
    lib: Tuple[Tuple[str, Optional[str]], ...] = field(init=False, default=())
    model_supplement: str = field(init=False, default="")
    width: float = field(init=False, default=1, metadata={"unit": "micron"})
    nfing: int = field(init=False, default=1)
    length_precision: float = field(
        init=False, default=0.005, metadata={"unit": "micron"}
    )
    width_precision: float = field(
        init=False, default=0.005, metadata={"unit": "micron"}
    )
    min_length: float = field(init=False, metadata={"unit": "micron"})
    min_width: float = field(init=False, metadata={"unit": "micron"})
    max_length: float = field(init=False, metadata={"unit": "micron"})
    max_width: float = field(init=False, metadata={"unit": "micron"})
    vdd: float = field(init=False, metadata={"unit": "V"})

    # Simulator parameters
    simulator: Callable[["SweepConfig"], Simulator] = field(init=False, repr=False)
    simulator_options: Tuple[Tuple[str, ...], ...] = field(init=False, default=())
    sweep_parallel: bool = field(init=False, default=False)
    debug: bool = field(init=False, default=False)

    # Output variables
    _valid: set[str] = field(init=False, default_factory=set)
    _outvars: dict[str, dict[str, int]] | None = field(init=False, default=None)

    OUTVARS: Tuple[str, ...] = (
        "ID",
        "VT",
        "IGD",
        "IGS",
        "GM",
        "GMB",
        "GDS",
        "CGG",
        "CGS",
        "CSG",
        "CGD",
        "CDG",
        "CGB",
        "CDD",
        "CSS",
    )
    OUTVARS_NOISE: Tuple[str, ...] = (
        "STH",
        "SFL",
    )

    def __post_init__(self):
        self.savefile = Path(self.savefile)
        if not self.savefile.suffix:
            self.savefile = self.savefile.with_suffix(".h5")
        assert (
            self.savefile.suffix in LUTS.keys()
        ), f"Unsupported savefile format: {self.savefile.suffix}"

        # self.include = tuple(os.path.expandvars(inc) for inc in self.include)
        # self.lib = tuple(
        #     (os.path.expandvars(lib), section) for lib, section in self.lib
        # )

        if isinstance(self.temp, str):
            self.temp = convert_temp(self.temp)

        def div_arr(n, d):
            return np.isclose(n / d, np.array([np.floor(n / d), np.ceil(n / d)]))

        def is_divisible(n, d):
            return np.any(div_arr(n, d))

        for key in ("vds", "vgs", "vsb"):
            value = getattr(self, key)
            setattr(self, f"{key}_max", max(value))
            setattr(self, f"{key}_min", min(value))
            setattr(self, f"{key}_step", np.round(value[1] - value[0], 6))

            range = getattr(self, f"{key}_max") - getattr(self, f"{key}_min")

            assert is_divisible(
                range, getattr(self, f"{key}_step")
            ), f"{key.upper()} Range ({range}) must be divisible by step size ({getattr(self, f'{key}_step')}) ({div_arr(range, getattr(self, f'{key}_step'))})"

        self.length_vec = (
            np.round(np.array(self.length) / self.length_precision)
            * self.length_precision
        )

    def __iter__(self):
        key_mapping = {
            "info": "INFO",
            "corner": "CORNER",
            "include": "INCLUDE",
            "lib": "LIB",
            "model": "MODEL",
            "device": "DEVICE",
            "vdd": "VDD",
            "temp": "TEMP",
            "vgs": "VGS",
            "vds": "VDS",
            "vsb": "VSB",
            "length": "L",
            "width": "W",
            "nfing": "NFING",
            "min_width": "MIN_WIDTH",
            "min_length": "MIN_LENGTH",
            "max_width": "MAX_WIDTH",
            "max_length": "MAX_LENGTH",
            "width_precision": "WIDTH_PRECISION",
            "length_precision": "LENGTH_PRECISION",
            # "simulator_options": "SIMULATOR_OPTIONS",
        }
        for cfg_key, lut_key in key_mapping.items():
            value = getattr(self, cfg_key)
            if isinstance(value, tuple) and all(
                isinstance(v, numbers.Number) for v in value
            ):
                value = np.array(value)
            yield lut_key, value

    @cached_property
    def lut(self) -> _BaseLUT:
        return LUTS[Path(self.savefile).suffix]

    @property
    def outvar_mapping(self) -> Dict[str, Dict[str, int | Callable[[float], float]]]:
        """Return the mapping of output variables from the simulation to the lookup table.

        outvars: `['ID','VT','IGD','IGS','GM','GMB','GDS','CGG','CGS','CSG','CGD','CDG','CGB','CDD','CSS']`
        outvars_noise: `['STH','SFL']`

        """
        if result := getattr(self, "_outvars", None):
            return result

        if not hasattr(self, "_valid"):
            self._valid = set(self.OUTVARS + self.OUTVARS_NOISE)

        result = dict(
            filter(
                lambda it: any(out in self._valid for out in it[1].keys()),
                self._get_outvar_mapping().items(),
            )
        )
        if len(result) == 0:
            LOGGER.warning("No valid output variable mapping found")

        assert all(
            all(callable(v) or v in [-1, 0, 1] for v in result[key].values())
            for key in result
        ), "All output variable values must be in [-1, 0, 1]"
        self._outvars = result
        return result

    @outvar_mapping.setter
    def outvar_mapping(self, valid_keys: Iterable[str]):
        self._outvars = None
        self._valid = set(valid_keys)

    @abstractmethod
    def _get_outvar_mapping(self) -> Dict[str, Dict[str, int]]:
        """Return the mapping of output variables from the simulation to the lookup table."""
        pass
