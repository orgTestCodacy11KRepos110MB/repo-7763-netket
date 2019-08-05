from __future__ import absolute_import
from .._C_netket.machine import *
from .cxx_machine import *


def _has_jax():
    try:
        import jax

        return True
    except ImportError:
        return False


if _has_jax():
    from .jax import *


def MPSPeriodicDiagonal(hilbert, bond_dim, symperiod=-1):
    return MPSPeriodic(hilbert, bond_dim, diag=True, symperiod=symperiod)
