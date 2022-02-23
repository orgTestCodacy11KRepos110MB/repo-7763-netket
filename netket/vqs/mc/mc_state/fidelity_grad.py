from functools import partial
from typing import Any

import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket.stats import Stats, statistics, mean
from netket.utils.dispatch import dispatch
from netket.utils import mpi

from netket.vqs.mc import check_hilbert

from .state import MCState


@dispatch
def fidelity_and_grad(ψ: MCState, ϕ: MCState, mutable: Any):  # noqa: F811
    check_hilbert(ψ.hilbert, ϕ.hilbert)

    σ_ψ = ψ.samples.reshape((-1, ψ.hilbert.size))
    σ_Φ = ϕ.samples.reshape((-1, ϕ.hilbert.size))

    return _fidelity_mcstates(
        ψ._apply_fun,
        ψ.parameters,
        ψ.model_state,
        σ_ψ,
        ϕ._apply_fun,
        ϕ.parameters,
        ϕ.model_state,
        σ_Φ,
    )


@partial(jax.jit, static_argnums=(0, 4))
def _fidelity_mcstates(logψ, w_ψ, ms_ψ, σ_ψ, logϕ, w_ϕ, ms_ϕ, σ_ϕ):
    def _logψ(w_ψ):
        return logψ({"params": w_ψ, **ms_ψ}, σ_ψ)

    vars_ψ = {"params": w_ψ, **ms_ψ}
    vars_ϕ = {"params": w_ϕ, **ms_ϕ}

    ψ_over_Φ = jnp.exp(logψ(vars_ψ, σ_ϕ) - logϕ(vars_ϕ, σ_ϕ))
    Φ_over_Ψ = jnp.exp(logϕ(vars_ϕ, σ_ψ) - logψ(vars_ψ, σ_ψ))

    E_ψ_over_Φ = mean(ψ_over_Φ)
    E_Φ_over_Ψ = mean(Φ_over_Ψ)

    fidelity = E_ψ_over_Φ * E_Φ_over_Ψ

    ## gradient
    Φ_over_Ψ_centered = Φ_over_Ψ - E_Φ_over_Ψ

    logψ, logψ_vjp = nkjax.vjp(_logψ, w_ψ)
    O_k_Φ_over_Ψ_centered = logψ_vjp(Φ_over_Ψ_centered)

    fidelity_grad = jax.tree_map(lambda x: x * E_ψ_over_Φ, O_k_Φ_over_Ψ_centered)

    return fidelity, fidelity_grad
