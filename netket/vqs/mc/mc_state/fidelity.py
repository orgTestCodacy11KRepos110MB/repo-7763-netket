from functools import partial

import jax
from jax import numpy as jnp

from netket.stats import Stats, mean, statistics
from netket.utils.dispatch import dispatch

from netket.vqs.mc import check_hilbert
from .state import MCState


@dispatch
def fidelity(ψ: MCState, ϕ: MCState):  # noqa: F811
    check_hilbert(ψ.hilbert, ϕ.hilbert)

    σ_ψ = ψ.samples.reshape((-1, ψ.hilbert.size))
    σ_Φ = ϕ.samples.reshape((-1, ϕ.hilbert.size))

    return _fidelity_mcstates(
        ψ._apply_fun, ψ.variables, σ_ψ, ϕ._apply_fun, ϕ.variables, σ_Φ
    )


@partial(jax.jit, static_argnums=(0, 3))
def _fidelity_mcstates(logψ, w_ψ, σ_ψ, logϕ, w_ϕ, σ_ϕ):

    ψ_over_Φ = jnp.exp(logψ(w_ψ, σ_ϕ) - logϕ(w_ϕ, σ_ϕ))
    Φ_over_Ψ = jnp.exp(logϕ(w_ϕ, σ_ψ) - logψ(w_ψ, σ_ψ))

    E_ψ_over_Φ = mean(ψ_over_Φ)
    E_Φ_over_Ψ = mean(Φ_over_Ψ)

    return E_ψ_over_Φ * E_Φ_over_Ψ
