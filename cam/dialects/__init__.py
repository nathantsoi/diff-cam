# SPDX-License-Identifier: MIT
"""Clean-room multi-controller CNC G-code dialect specifications and registry."""

from __future__ import annotations

from benchmarks.contracts import Dialect

from .base import (
    INDUSTRIAL_EIA_DIALECTS,
    ControllerDialect,
    ControllerDialectSpec,
    HardwareSyntaxError,
    coerce_controller_dialect,
    fmt_int,
    fmt_real,
    is_dialect_allowed_by_task,
)
from .fanuc_robodrill import (
    FANUC_SPEC,
    ROBODRILL_SPEC,
    FanucDialectSpec,
    RobodrillDialectSpec,
)
from .haas_linuxcnc import (
    HAAS_NGC_SPEC,
    LINUXCNC_RS274_SPEC,
    HaasNGCDialectSpec,
    LinuxCNCRS274DialectSpec,
    format_haas_ngc_shop_program,
)
from .makino_mazak import (
    MAKINO_SPEC,
    MAZAK_SPEC,
    MakinoDialectSpec,
    MazakDialectSpec,
)

DIALECT_SPECS: dict[ControllerDialect, ControllerDialectSpec] = {
    ControllerDialect.RS274: LINUXCNC_RS274_SPEC,
    ControllerDialect.HAAS: HAAS_NGC_SPEC,
    ControllerDialect.FANUC: FANUC_SPEC,
    ControllerDialect.ROBODRILL: ROBODRILL_SPEC,
    ControllerDialect.MAKINO: MAKINO_SPEC,
    ControllerDialect.MAZAK: MAZAK_SPEC,
}


def get_dialect_spec(dialect: Dialect | ControllerDialect | str) -> ControllerDialectSpec:
    """Return the singleton :class:`ControllerDialectSpec` for ``dialect``."""
    ctrl = coerce_controller_dialect(dialect)
    return DIALECT_SPECS[ctrl]


def verify_hardware_syntax(
    nc_text: str,
    dialect: Dialect | ControllerDialect | str,
    *,
    units: str = "mm",
    shop_mode: bool = False,
) -> None:
    """Validate ``nc_text`` against ``dialect``'s physical controller syntax rules."""
    get_dialect_spec(dialect).verify_hardware_syntax(
        nc_text, units=units, shop_mode=shop_mode
    )


__all__ = [
    "ControllerDialect",
    "ControllerDialectSpec",
    "DIALECT_SPECS",
    "FANUC_SPEC",
    "FanucDialectSpec",
    "HAAS_NGC_SPEC",
    "HaasNGCDialectSpec",
    "HardwareSyntaxError",
    "INDUSTRIAL_EIA_DIALECTS",
    "LINUXCNC_RS274_SPEC",
    "LinuxCNCRS274DialectSpec",
    "MAKINO_SPEC",
    "MAZAK_SPEC",
    "MakinoDialectSpec",
    "MazakDialectSpec",
    "ROBODRILL_SPEC",
    "RobodrillDialectSpec",
    "coerce_controller_dialect",
    "fmt_int",
    "fmt_real",
    "format_haas_ngc_shop_program",
    "get_dialect_spec",
    "is_dialect_allowed_by_task",
    "verify_hardware_syntax",
]
