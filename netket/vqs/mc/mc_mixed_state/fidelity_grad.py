from functools import partial
from typing import Any

import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket.stats import Stats, statistics, mean
from netket.utils.dispatch import dispatch
from netket.utils import mpi

from netket.vqs.mc import check_hilbert
from netket.vqs.mc import MCState
from .state import MCMixedState

# computes ⟨ψ|ρ|ψ⟩
@dispatch
def fidelity_and_grad(ψ: MCState, ρ: MCMixedState, mutable: Any):  # noqa: F811
    check_hilbert(ψ.hilbert, ρ.hilbert.physical)

    return _fidelity_pure_mixed_mcstates(
        ψ._apply_fun,
        ψ.parameters,
        ψ.model_state,
        ρ._apply_fun,
        ρ.parameters,
        ρ.model_state,
        ρ.sample,
    )


@partial(jax.jit, static_argnums=(0, 3))
def _fidelity_pure_mixed_mcstates(logψ, w_ψ, ms_ψ, logρ, w_ρ, ms_ρ, ση_ρ):

    shape = ση_ρ.shape
    N2 = ση_ρ.shape[-1]
    N = N2 // 2

    σ = (ση_ρ[..., 0:N]).reshape(-1, N)
    η = (ση_ρ[..., N:]).reshape(-1, N)

    def _logψ(pars, x):
        return logψ({"params": w_ψ, **ms_ψ}, x)

    def _logρ(pars, x):
        return logρ({"params": w_ρ, **ms_ρ}, x)

    logρ_ση = logρ(w_ρ, ση_ρ)

    logψ_σ, vjp_σ = nkjax.vjp(_logψ, w_ψ, σ)
    logψ_η, vjp_η = nkjax.vjp(_logψ, w_ψ, η)

    fidelity = jnp.exp(logψ_σ.conj() + logψ_η - logρ_ση.conj())

    grad_σ = vjp_σ(fidelity)[0]
    grad_η = vjp_η(fidelity)[0]

    grad = jax.tree_multimap(lambda x, y: x.conj() + y, grad_σ, grad_η)
    grad = jax.tree_map(lambda x: mpi.mpi_mean_jax(-x.conj()), grad)

    F = statistics(fidelity.real.reshape(shape[:-1]).T)

    return F, grad
