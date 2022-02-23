from functools import partial

import jax
from jax import numpy as jnp

from netket.stats import Stats, mean, statistics
from netket.utils.dispatch import dispatch

from netket.vqs.mc import MCState
from .state import MCMixedState

from netket.vqs.mc import check_hilbert

# computes ⟨ψ|ρ|ψ⟩
@dispatch
def fidelity(ψ: MCState, ρ: MCMixedState):  # noqa: F811
    check_hilbert(ψ.hilbert, ρ.hilbert.physical)

    return _fidelity_pure_mixed_mcstates(
        ψ._apply_fun, ψ.variables, ρ._apply_fun, ρ.variables, ρ.sample
    )


@partial(jax.jit, static_argnums=(0, 3))
def _fidelity_pure_mixed_mcstates(logψ, w_ψ, logρ, w_ρ, ση_ρ):

    shape = ση_ρ.shape
    N2 = ση_ρ.shape[-1]
    N = N2 // 2

    σ = (ση_ρ[..., 0:N]).reshape(-1, N)
    η = (ση_ρ[..., N:]).reshape(-1, N)

    logρ_ση = logρ(w_ρ, ση_ρ)
    logψ_σ = logψ(w_ψ, σ)
    logψ_η = logψ(w_ψ, η)

    fidelity = jnp.exp(logψ_σ + logψ_η.conj() - logρ_ση.conj())
    F = statistics(fidelity.real.reshape(shape[:-1]).T)

    return F
