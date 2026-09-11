from __future__ import annotations

import numpy as np

from co2ie.conformers import Xyz


def combine_host_co2(host: Xyz, co2: Xyz, comment: str = "host+CO2") -> Xyz:
    """Concatenate host then CO2 frames (host atoms first, then C,O,O)."""
    return Xyz(
        symbols=list(host.symbols) + list(co2.symbols),
        coords=np.vstack([host.coords, co2.coords]),
        comment=comment,
    )
