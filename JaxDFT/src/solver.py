"""Self-consistent field solver for real-space Kohn-Sham DFT in JAX.

All quantities are in atomic units: length in Bohr, energy in Hartree, and
forces in Hartree/Bohr. The SCF loop builds the effective potential, solves
the Kohn-Sham eigenproblem, and mixes the density until convergence.
"""

import jax
import jax.numpy as jnp
import numpy as np
from .backends.uniform import (
    UniformBackend,
    precompute_uniform_poisson_kernel,
    solve_uniform_poisson,
)
from .functional import lda_xc


def _dirichlet_project_field(grid, field):
    """Project a field onto the grid's Dirichlet mask when present."""
    mask = getattr(grid, "mask", None)
    if mask is None:
        return field
    field = jnp.asarray(field)
    return field * jnp.asarray(mask, dtype=field.dtype)


def _dirichlet_project_block(grid, block):
    """Project each flattened column vector onto the grid's Dirichlet mask."""
    mask = getattr(grid, "mask", None)
    if mask is None:
        return block
    block = jnp.asarray(block)
    mask_flat = jnp.asarray(mask, dtype=block.dtype).reshape((-1, 1))
    return block * mask_flat


def _maybe_trace_append(trace_sink, payload):
    """Append a payload to a trace sink when tracing is enabled."""
    if trace_sink is not None:
        trace_sink.append(payload)


def _trace_array(arr):
    """Convert a JAX array to a compact NumPy array for Python-side traces."""
    return np.asarray(jnp.asarray(arr), dtype=np.float32)


def _rms_norm(arr):
    """Return the RMS norm of an array-like object as a Python float."""
    arr = jnp.asarray(arr, dtype=jnp.float32)
    return float(jnp.sqrt(jnp.mean(arr * arr)))


def _state0_overlap_prev(grid, backend, prev_block, curr_block):
    """Return the overlap between the previous and current leading state."""
    if prev_block is None or curr_block is None:
        return None
    if prev_block.shape[1] == 0 or curr_block.shape[1] == 0:
        return None
    prev0 = prev_block[:, 0].reshape(grid.shape)
    curr0 = curr_block[:, 0].reshape(grid.shape)
    prev_norm = float(backend.inner_product(grid, prev0, prev0))
    curr_norm = float(backend.inner_product(grid, curr0, curr0))
    if prev_norm <= 0.0 or curr_norm <= 0.0:
        return None
    return float(jnp.abs(backend.inner_product(grid, prev0, curr0)))


def _occupied_subspace_overlap_prev(grid, backend, prev_block, curr_block, max_m=2):
    """Return a small occupied-subspace continuity metric across iterations."""
    if prev_block is None or curr_block is None:
        return None
    m = min(int(prev_block.shape[1]), int(curr_block.shape[1]), int(max_m))
    if m <= 0:
        return None
    prev_fields = jnp.moveaxis(prev_block[:, :m].reshape(grid.shape + (m,)), -1, 0)
    curr_fields = jnp.moveaxis(curr_block[:, :m].reshape(grid.shape + (m,)), -1, 0)
    gram = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            gram[i, j] = float(backend.inner_product(grid, curr_fields[i], prev_fields[j]))
    sigma = np.linalg.svd(gram, compute_uv=False)
    sigma = np.clip(sigma, 0.0, 1.0)
    return float(np.min(sigma))


def _metric_scale(grid, backend):
    """Return a normalization scale for backend-aware inner products.

    The scale is chosen so that a uniform grid with cell volume ``dv`` reduces
    to the usual Euclidean dot product up to numerical noise.
    """
    ones = jnp.ones(grid.shape, dtype=jnp.float32)
    n_grid = jnp.asarray(jnp.prod(jnp.asarray(grid.shape)), dtype=jnp.float32)
    return backend.integrate(grid, ones) / jnp.maximum(n_grid, 1.0)


def _metric_inner_flat(grid, backend, x, y, scale):
    """Evaluate the backend-aware inner product for flattened vectors."""
    x_field = x.reshape(grid.shape)
    y_field = y.reshape(grid.shape)
    return backend.inner_product(grid, x_field, y_field) / scale


def _metric_gram(grid, backend, X, Y, scale):
    """Build the backend-aware block Gram matrix X^T W Y."""

    def against_y(y_col):
        return jax.vmap(lambda x_col: _metric_inner_flat(grid, backend, x_col, y_col, scale), in_axes=1)(X)

    return jax.vmap(against_y, in_axes=1, out_axes=1)(Y)


def _metric_symmetrize(mat):
    """Symmetrize a small dense matrix to reduce numerical asymmetry."""
    return 0.5 * (mat + mat.T)


def _metric_orthonormalize_with_transform(grid, backend, X, scale, eps=1e-8):
    """Metric-orthonormalize a block and return the linear transform used."""
    S = _metric_symmetrize(_metric_gram(grid, backend, X, X, scale))
    evals, evecs = jnp.linalg.eigh(S)
    eps = jnp.asarray(eps, dtype=evals.dtype)
    safe_evals = jnp.maximum(evals, eps)
    inv_sqrt = 1.0 / jnp.sqrt(safe_evals)
    transform = evecs * inv_sqrt[None, :]
    return X @ transform, transform


def _metric_orthonormalize(grid, backend, X, scale, eps=1e-8):
    """Metric-orthonormalize a block of vectors using its small Gram matrix."""
    X_ortho, _ = _metric_orthonormalize_with_transform(grid, backend, X, scale, eps=eps)
    return X_ortho


def _metric_project_out(grid, backend, basis, vecs, scale):
    """Project vecs orthogonally to basis under the backend-aware metric."""
    coeff = _metric_gram(grid, backend, basis, vecs, scale)
    return vecs - basis @ coeff


def _metric_block_norms(grid, backend, R, scale):
    """Return backend-aware norms for each column in R."""

    def one_norm(vec):
        val = _metric_inner_flat(grid, backend, vec, vec, scale)
        return jnp.sqrt(jnp.maximum(val, 0.0))

    return jax.vmap(one_norm, in_axes=1, out_axes=0)(R)


def solve_orbitals_subspace(
    apply_h_fn,
    n_grid,
    n_bands,
    x_init=None,
    max_iter=100,
    tol=1e-5,
    key=None,
    grid=None,
    backend=None,
    trace_sink=None,
    trace_context=None,
):
    """
    Iterative eigensolver using block subspace expansion + Rayleigh-Ritz.

    Stops when the maximum band residual
        max_i ||H psi_i - eps_i psi_i||_2
    falls below tol.

    When ``grid`` and ``backend`` are both provided, the solver uses the
    backend-aware weighted metric defined by ``backend.inner_product``.
    Otherwise it falls back to the original Euclidean implementation.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    if (grid is None) != (backend is None):
        raise ValueError(
            "solve_orbitals_subspace expects both grid and backend together "
            "for the metric-aware path"
        )

    metric_enabled = grid is not None and backend is not None
    trace_enabled = trace_sink is not None
    if trace_context is None:
        trace_context = {}

    if x_init is None:
        X = jax.random.normal(key, (n_grid, n_bands)).astype(jnp.float32)
    else:
        X = x_init + 1e-6 * jax.random.normal(key, (n_grid, n_bands)).astype(jnp.float32)

    if not metric_enabled:
        # Initial orthonormalization
        X = jnp.linalg.qr(X)[0]
        HX = jax.vmap(apply_h_fn, in_axes=1, out_axes=1)(X)

        def cond_fun(state):
            i, _, _, _, res_norm = state
            return jnp.logical_and(i < max_iter, res_norm > tol)

        def body_fun(state):
            i, X, HX, _, _ = state

            # 1) Rayleigh-Ritz in current subspace
            H_sub = X.T @ HX
            H_sub = 0.5 * (H_sub + H_sub.T)   # numerical symmetrization
            E, V_sub = jnp.linalg.eigh(H_sub)

            X = X @ V_sub
            HX = HX @ V_sub

            # 2) Residual in current subspace
            R = HX - X * E[None, :]
            res_vec = jnp.sqrt(jnp.sum(R * R, axis=0))
            res_norm = jnp.max(res_vec)

            def done_branch(_):
                return i + 1, X, HX, E, res_norm

            def expand_branch(_):
                # 3) Orthogonalize residuals against current subspace
                R_ortho = R - X @ (X.T @ R)
                R_ortho = jnp.linalg.qr(R_ortho)[0]

                # 4) Expand subspace
                HR = jax.vmap(apply_h_fn, in_axes=1, out_axes=1)(R_ortho)

                Z = jnp.concatenate([X, R_ortho], axis=1)
                HZ = jnp.concatenate([HX, HR], axis=1)

                # 5) Rayleigh-Ritz in expanded subspace
                H_Z = Z.T @ HZ
                H_Z = 0.5 * (H_Z + H_Z.T)
                E_Z, V_Z = jnp.linalg.eigh(H_Z)

                X_new = Z @ V_Z[:, :n_bands]
                HX_new = HZ @ V_Z[:, :n_bands]
                E_new = E_Z[:n_bands]

                # 6) Residual after update
                R_new = HX_new - X_new * E_new[None, :]
                res_vec_new = jnp.sqrt(jnp.sum(R_new * R_new, axis=0))
                res_norm_new = jnp.max(res_vec_new)

                return i + 1, X_new, HX_new, E_new, res_norm_new

            return jax.lax.cond(res_norm <= tol, done_branch, expand_branch, operand=None)

        state0 = (
            jnp.array(0, dtype=jnp.int32),
            X,
            HX,
            jnp.full((n_bands,), jnp.inf, dtype=jnp.float32),
            jnp.array(jnp.inf, dtype=jnp.float32),
        )

        _, X_final, _, E_final, _ = jax.lax.while_loop(cond_fun, body_fun, state0)
        return E_final, X_final

    scale = _metric_scale(grid, backend)
    X = _dirichlet_project_block(grid, X)
    X = _metric_orthonormalize(grid, backend, X, scale)
    X = _dirichlet_project_block(grid, X)
    HX = jax.vmap(apply_h_fn, in_axes=1, out_axes=1)(X)
    HX = _dirichlet_project_block(grid, HX)

    if not trace_enabled:
        def cond_fun(state):
            i, _, _, _, res_norm = state
            return jnp.logical_and(i < max_iter, res_norm > tol)

        def body_fun(state):
            i, X, HX, _, _ = state

            # 1) Rayleigh-Ritz in current metric-orthonormal subspace
            H_sub = _metric_symmetrize(_metric_gram(grid, backend, X, HX, scale))
            E, V_sub = jnp.linalg.eigh(H_sub)

            X = X @ V_sub
            HX = HX @ V_sub
            X = _dirichlet_project_block(grid, X)
            HX = _dirichlet_project_block(grid, HX)

            # 2) Residual in current subspace
            R = HX - X * E[None, :]
            R = _dirichlet_project_block(grid, R)
            res_vec = _metric_block_norms(grid, backend, R, scale)
            res_norm = jnp.max(res_vec)

            def done_branch(_):
                return i + 1, X, HX, E, res_norm

            def expand_branch(_):
                # 3) Orthogonalize residuals against current subspace in metric W
                R_ortho = _metric_project_out(grid, backend, X, R, scale)
                R_ortho = _dirichlet_project_block(grid, R_ortho)
                R_ortho = _metric_orthonormalize(grid, backend, R_ortho, scale)
                R_ortho = _dirichlet_project_block(grid, R_ortho)

                # 4) Expand and re-orthonormalize the full subspace for stability
                Z0 = jnp.concatenate([X, R_ortho], axis=1)
                Z0 = _dirichlet_project_block(grid, Z0)
                Z = _metric_orthonormalize(grid, backend, Z0, scale)
                Z = _dirichlet_project_block(grid, Z)
                HZ = jax.vmap(apply_h_fn, in_axes=1, out_axes=1)(Z)
                HZ = _dirichlet_project_block(grid, HZ)

                # 5) Rayleigh-Ritz in expanded metric-orthonormal subspace
                H_Z = _metric_symmetrize(_metric_gram(grid, backend, Z, HZ, scale))
                E_Z, V_Z = jnp.linalg.eigh(H_Z)

                X_new = Z @ V_Z[:, :n_bands]
                HX_new = HZ @ V_Z[:, :n_bands]
                X_new = _dirichlet_project_block(grid, X_new)
                HX_new = _dirichlet_project_block(grid, HX_new)
                E_new = E_Z[:n_bands]

                # 6) Residual after update
                R_new = HX_new - X_new * E_new[None, :]
                R_new = _dirichlet_project_block(grid, R_new)
                res_vec_new = _metric_block_norms(grid, backend, R_new, scale)
                res_norm_new = jnp.max(res_vec_new)

                return i + 1, X_new, HX_new, E_new, res_norm_new

            return jax.lax.cond(res_norm <= tol, done_branch, expand_branch, operand=None)

        state0 = (
            jnp.array(0, dtype=jnp.int32),
            X,
            HX,
            jnp.full((n_bands,), jnp.inf, dtype=jnp.float32),
            jnp.array(jnp.inf, dtype=jnp.float32),
        )
        _, X_final, _, E_final, _ = jax.lax.while_loop(cond_fun, body_fun, state0)
        return E_final, X_final

    E = jnp.full((n_bands,), jnp.inf, dtype=jnp.float32)
    res_norm = jnp.array(jnp.inf, dtype=jnp.float32)
    i = 0

    while i < max_iter and float(res_norm) > tol:
        current_dim = int(X.shape[1])

        # 1) Rayleigh-Ritz in current metric-orthonormal subspace
        H_sub = _metric_symmetrize(_metric_gram(grid, backend, X, HX, scale))
        E, V_sub = jnp.linalg.eigh(H_sub)

        X = X @ V_sub
        HX = HX @ V_sub
        X = _dirichlet_project_block(grid, X)
        HX = _dirichlet_project_block(grid, HX)

        # 2) Residual in current subspace
        R = HX - X * E[None, :]
        R = _dirichlet_project_block(grid, R)
        res_vec = _metric_block_norms(grid, backend, R, scale)
        res_norm = jnp.max(res_vec)
        _maybe_trace_append(
            trace_sink,
            {
                "scf_iter": trace_context.get("scf_iter"),
                "sub_iter": int(i),
                "stage": "rayleigh_ritz",
                "subspace_dim": current_dim,
                "expanded_subspace_dim": current_dim * 2,
                "ritz_eigvals": _trace_array(E[: min(4, E.shape[0])]),
                "max_residual_norm": float(res_norm),
            },
        )
        if float(res_norm) <= tol:
            i += 1
            break

        # 3) Orthogonalize residuals against current subspace in metric W
        R_ortho = _metric_project_out(grid, backend, X, R, scale)
        R_ortho = _dirichlet_project_block(grid, R_ortho)
        R_ortho = _metric_orthonormalize(grid, backend, R_ortho, scale)
        R_ortho = _dirichlet_project_block(grid, R_ortho)

        # 4) Expand and re-orthonormalize the full subspace for stability
        Z0 = jnp.concatenate([X, R_ortho], axis=1)
        Z0 = _dirichlet_project_block(grid, Z0)
        Z = _metric_orthonormalize(grid, backend, Z0, scale)
        Z = _dirichlet_project_block(grid, Z)
        HZ = jax.vmap(apply_h_fn, in_axes=1, out_axes=1)(Z)
        HZ = _dirichlet_project_block(grid, HZ)

        # 5) Rayleigh-Ritz in expanded metric-orthonormal subspace
        H_Z = _metric_symmetrize(_metric_gram(grid, backend, Z, HZ, scale))
        E_Z, V_Z = jnp.linalg.eigh(H_Z)

        X = Z @ V_Z[:, :n_bands]
        HX = HZ @ V_Z[:, :n_bands]
        X = _dirichlet_project_block(grid, X)
        HX = _dirichlet_project_block(grid, HX)
        E = E_Z[:n_bands]

        # 6) Residual after update
        R_new = HX - X * E[None, :]
        R_new = _dirichlet_project_block(grid, R_new)
        res_vec_new = _metric_block_norms(grid, backend, R_new, scale)
        res_norm = jnp.max(res_vec_new)
        _maybe_trace_append(
            trace_sink,
            {
                "scf_iter": trace_context.get("scf_iter"),
                "sub_iter": int(i),
                "stage": "expanded_update",
                "subspace_dim": int(Z.shape[1]),
                "expanded_subspace_dim": int(Z.shape[1]),
                "ritz_eigvals": _trace_array(E[: min(4, E.shape[0])]),
                "max_residual_norm": float(res_norm),
            },
        )
        i += 1

    return E, X

def solve_orbitals_lobpcg(*args, **kwargs):
    """
    Backward-compatible alias.

    The current implementation is a block subspace expansion solver,
    not a strict textbook LOBPCG.
    """
    return solve_orbitals_subspace(*args, **kwargs)


def _resolve_backend(backend):
    return UniformBackend() if backend is None else backend


def _resolve_adaptive_eigensolver_schedule(
    backend,
    adaptive_scf_aware_eigensolver=None,
    adaptive_eigensolver_early_max_iter=None,
    adaptive_eigensolver_late_max_iter=None,
    adaptive_eigensolver_stage_residual_threshold=None,
    adaptive_eigensolver_late_tol=None,
):
    backend_name = getattr(backend, "name", None)
    legacy_max_iter = 8 if backend_name == "adaptive_tensor" else 30
    legacy_tol = 1.0e-4 if backend_name == "adaptive_tensor" else 1.0e-5
    if backend_name != "adaptive_tensor":
        return {
            "enabled": False,
            "legacy_max_iter": legacy_max_iter,
            "legacy_tol": legacy_tol,
            "early_max_iter": None,
            "late_max_iter": None,
            "stage_threshold": None,
            "late_tol": None,
        }
    enabled = getattr(backend, "adaptive_scf_aware_eigensolver", False)
    if adaptive_scf_aware_eigensolver is not None:
        enabled = bool(adaptive_scf_aware_eigensolver)
    early_max_iter = getattr(backend, "adaptive_eigensolver_early_max_iter", 4)
    if adaptive_eigensolver_early_max_iter is not None:
        early_max_iter = adaptive_eigensolver_early_max_iter
    late_max_iter = getattr(backend, "adaptive_eigensolver_late_max_iter", 12)
    if adaptive_eigensolver_late_max_iter is not None:
        late_max_iter = adaptive_eigensolver_late_max_iter
    stage_threshold = getattr(backend, "adaptive_eigensolver_stage_residual_threshold", 1.0e-3)
    if adaptive_eigensolver_stage_residual_threshold is not None:
        stage_threshold = adaptive_eigensolver_stage_residual_threshold
    late_tol = getattr(backend, "adaptive_eigensolver_late_tol", 1.0e-5)
    if adaptive_eigensolver_late_tol is not None:
        late_tol = adaptive_eigensolver_late_tol
    return {
        "enabled": bool(enabled),
        "legacy_max_iter": int(legacy_max_iter),
        "legacy_tol": float(legacy_tol),
        "early_max_iter": int(early_max_iter),
        "late_max_iter": int(late_max_iter),
        "stage_threshold": float(stage_threshold),
        "late_tol": float(late_tol),
    }


def _choose_eigensolver_budget(schedule, density_residual_metric):
    if not schedule["enabled"]:
        return "fixed", schedule["legacy_max_iter"], schedule["legacy_tol"]
    residual_value = float(density_residual_metric)
    if not np.isfinite(residual_value) or residual_value > schedule["stage_threshold"]:
        return "early", schedule["early_max_iter"], schedule["legacy_tol"]
    return "late", schedule["late_max_iter"], schedule["late_tol"]


def _summarize_orbital_trace(orbital_trace, n_bands):
    if not orbital_trace:
        return None
    iterations = sum(1 for event in orbital_trace if event.get("stage") == "rayleigh_ritz")
    hpsi_calls = int(n_bands)
    for event in orbital_trace:
        if event.get("stage") == "expanded_update":
            hpsi_calls += int(event.get("subspace_dim", 0))
    return {
        "iterations": int(iterations),
        "final_residual": float(orbital_trace[-1].get("max_residual_norm", np.nan)),
        "final_subspace_dim": int(orbital_trace[-1].get("subspace_dim", n_bands)),
        "hpsi_calls": int(hpsi_calls),
    }


def precompute_poisson_kernel(grid_shape, spacing):
    """Backward-compatible wrapper for the current uniform Poisson kernel."""
    return precompute_uniform_poisson_kernel(grid_shape, spacing)


def solve_poisson(rho, kernel_k, spacing):
    """Backward-compatible wrapper for the current uniform Poisson solve."""
    return solve_uniform_poisson(rho, kernel_k, spacing)


def solve_orbitals_dense(apply_h_fn, n_grid, n_bands):
    """Diagonalize the dense Hamiltonian to obtain Kohn-Sham orbitals.

    Args:
        apply_h_fn: Linear operator that applies H to a flattened wavefunction.
        n_grid: Total number of grid points (product of grid dimensions).
        n_bands: Number of lowest eigenpairs to return.

    Returns:
        Tuple (eigvals, eigvecs) with eigenvalues in Hartree and eigenvectors
        shaped (n_grid, n_bands).
    """
    # 【核心回归】使用 Dense Solver (eigh)
    # 对于 Grid=0.18 (N~20k)，矩阵仅 1.7GB，完全可控且绝对收敛
    eye = jnp.eye(n_grid, dtype=jnp.float32)
    h_dense = jax.vmap(apply_h_fn, in_axes=1, out_axes=1)(eye)
    # 确保对称性
    h_dense = 0.5 * (h_dense + h_dense.T)
    # 对角化
    eigvals, eigvecs = jnp.linalg.eigh(h_dense)
    return eigvals[:n_bands], eigvecs[:, :n_bands]


def anderson_mixing(rho, rho_new, f_hist, mix_alpha, iter_idx, m=5):
    """Perform Anderson mixing for density updates.

    Args:
        rho: Current density (flattened), in Bohr^-3.
        rho_new: New density (flattened), in Bohr^-3.
        f_hist: History buffer of residuals, shape (m, n_grid).
        mix_alpha: Linear mixing parameter.
        iter_idx: Current SCF iteration index.
        m: History length for Anderson mixing.

    Returns:
        Tuple (rho_next, f_hist_next) with mixed density and updated history.
    """
    f = rho_new - rho
    m_val = m
    def first(_):
        f_hist0 = f_hist.at[0].set(f)
        return rho + mix_alpha * f, f_hist0
    def later(_):
        mcur = jnp.minimum(iter_idx, m_val)
        f_hist1 = f_hist.at[iter_idx % m_val].set(f)
        indices = (iter_idx - jnp.arange(m_val)) % m_val
        f_stack = jnp.swapaxes(f_hist1[indices], 0, 1)
        f_last = f_stack[:, 0]
        mask = jnp.arange(m_val) < mcur
        f_stack = jnp.where(mask[None, :], f_stack, f_last[:, None])
        F = f_stack - f_last[:, None]
        B = F.T @ F
        rhs = F.T @ f_last
        coeff = jnp.linalg.solve(B + 1e-10 * jnp.eye(m_val), rhs)
        correction = F @ coeff
        rho_next = rho_new - mix_alpha * correction
        rho_next = jnp.nan_to_num(rho_next, nan=rho_new)
        return rho_next, f_hist1
    return jax.lax.cond(iter_idx == 0, first, later, operand=None)


def scf(
    grid,
    coords,
    n_bands,
    occ,
    V_loc,
    projectors,
    max_iter,
    mix_alpha,
    tolerance,
    key,
    backend=None,
    trace_sink=None,
    trace_mode="off",
    *,
    return_diagnostics=False,
    adaptive_scf_aware_eigensolver=None,
    adaptive_eigensolver_early_max_iter=None,
    adaptive_eigensolver_late_max_iter=None,
    adaptive_eigensolver_stage_residual_threshold=None,
    adaptive_eigensolver_late_tol=None,
):
    """Run the self-consistent field (SCF) loop."""
    backend = _resolve_backend(backend)
    coords = jnp.asarray(coords, dtype=jnp.float32)
    if key is None:
        key = jax.random.PRNGKey(42)
    trace_enabled = trace_sink is not None and trace_mode != "off" and getattr(backend, "name", None) != "uniform"
    diagnostics_enabled = bool(return_diagnostics)
    schedule = _resolve_adaptive_eigensolver_schedule(
        backend,
        adaptive_scf_aware_eigensolver=adaptive_scf_aware_eigensolver,
        adaptive_eigensolver_early_max_iter=adaptive_eigensolver_early_max_iter,
        adaptive_eigensolver_late_max_iter=adaptive_eigensolver_late_max_iter,
        adaptive_eigensolver_stage_residual_threshold=adaptive_eigensolver_stage_residual_threshold,
        adaptive_eigensolver_late_tol=adaptive_eigensolver_late_tol,
    )
    python_control_flow = trace_enabled or diagnostics_enabled or schedule["enabled"]

    rho = jnp.zeros(grid.shape, dtype=jnp.float32)
    for a in range(coords.shape[0]):
        r = jnp.linalg.norm(grid.coords - coords[a], axis=-1)
        rho = rho + jnp.exp(-2.0 * r**2)
    rho = _dirichlet_project_field(grid, rho)
    rho = rho / backend.integrate(grid, rho) * jnp.sum(occ)

    f_hist = jnp.zeros((5, rho.size), dtype=jnp.float32)
    n_grid = rho.size
    proj_data = backend.precompute_nonlocal(grid, coords, projectors)
    eigvals0 = jnp.zeros((n_bands,), dtype=jnp.float32)
    eigvecs0 = jnp.zeros((n_grid, n_bands), dtype=jnp.float32)
    V_H0 = jnp.zeros(grid.shape, dtype=jnp.float32)
    eps_xc0 = jnp.zeros(grid.shape, dtype=jnp.float32)
    v_xc0 = jnp.zeros(grid.shape, dtype=jnp.float32)
    diff0 = jnp.array(jnp.inf, dtype=jnp.float32)
    i0 = jnp.array(0, dtype=jnp.int32)

    scf_history = [] if diagnostics_enabled else None
    stage_counts = {"fixed": 0, "early": 0, "late": 0}
    eigensolver_total_inner_iterations = 0
    eigensolver_total_hpsi_calls = 0

    def cond(state):
        i, _, _, diff, _, _, _, _, _ = state
        return jnp.logical_and(i < max_iter, diff > tolerance)

    def body(state):
        nonlocal eigensolver_total_inner_iterations, eigensolver_total_hpsi_calls
        i, rho_cur, f_hist_cur, diff, _, eigvecs_prev, V_H_prev, _, _ = state
        rho_cur = jnp.clip(rho_cur, 1e-12, None)
        rho_cur = _dirichlet_project_field(grid, rho_cur)
        stage_name, orbital_max_iter, orbital_tol = _choose_eigensolver_budget(schedule, diff)
        V_H = backend.solve_hartree(grid, rho_cur, v_init=V_H_prev)
        eps_xc, v_xc = lda_xc(rho_cur)
        V_eff = V_loc + V_H + v_xc

        def apply_h(psi_flat):
            psi = psi_flat.reshape(grid.shape)
            psi = _dirichlet_project_field(grid, psi)
            kinetic_psi = backend.apply_kinetic(grid, psi)
            v_nonlocal = backend.apply_nonlocal(grid, psi, proj_data)
            hpsi = kinetic_psi + V_eff * psi + v_nonlocal
            hpsi = _dirichlet_project_field(grid, hpsi)
            return hpsi.reshape(-1)

        iter_key = jax.random.fold_in(key, i)
        metric_backend = backend if getattr(backend, "name", None) != "uniform" else None
        metric_grid = grid if metric_backend is not None else None
        orbital_trace = [] if metric_backend is not None and (trace_enabled or diagnostics_enabled) else None
        eigvals, eigvecs = solve_orbitals_subspace(
            apply_h,
            n_grid,
            n_bands,
            x_init=eigvecs_prev,
            max_iter=orbital_max_iter,
            tol=orbital_tol,
            key=iter_key,
            grid=metric_grid,
            backend=metric_backend,
            trace_sink=orbital_trace,
            trace_context={"scf_iter": int(i)} if orbital_trace is not None else None,
        )
        orbital_diag = _summarize_orbital_trace(orbital_trace, n_bands)

        eigvecs = _dirichlet_project_block(grid, eigvecs)
        eigvec_fields = jnp.moveaxis(eigvecs.reshape(grid.shape + (n_bands,)), -1, 0)
        norm = jnp.sqrt(jax.vmap(lambda psi: backend.inner_product(grid, psi, psi))(eigvec_fields))
        eigvecs = eigvecs / norm[None, :]
        eigvecs = _dirichlet_project_block(grid, eigvecs)

        rho_new = jnp.sum((eigvecs ** 2) * occ[None, :], axis=1).reshape(grid.shape)
        rho_new = _dirichlet_project_field(grid, rho_new)
        diff = jnp.max(jnp.abs(rho_new - rho_cur))
        rho_flat, f_hist_cur = anderson_mixing(rho_cur.reshape(-1), rho_new.reshape(-1), f_hist_cur, mix_alpha, i)
        rho_mixed = _dirichlet_project_field(grid, rho_flat.reshape(grid.shape))

        if diagnostics_enabled:
            stage_counts[stage_name] = stage_counts.get(stage_name, 0) + 1
            eig_inner_iters = None if orbital_diag is None else int(orbital_diag["iterations"])
            eig_final_residual = None if orbital_diag is None else float(orbital_diag["final_residual"])
            eig_hpsi_calls = None if orbital_diag is None else int(orbital_diag["hpsi_calls"])
            if orbital_diag is not None:
                eigensolver_total_inner_iterations += eig_inner_iters
                eigensolver_total_hpsi_calls += eig_hpsi_calls
            scf_history.append({
                "iter": int(i),
                "density_residual": float(diff),
                "converged_flag": bool(float(diff) <= tolerance),
                "eigensolver_stage": stage_name,
                "eigensolver_max_iter_budget": int(orbital_max_iter),
                "eigensolver_tol": float(orbital_tol),
                "eigensolver_inner_iterations": eig_inner_iters,
                "eigensolver_final_residual": eig_final_residual,
                "eigensolver_hpsi_calls": eig_hpsi_calls,
            })

        if trace_enabled:
            payload = {
                "scf_iter": int(i),
                "x_init": _trace_array(eigvecs_prev),
                "x_init_norm": float(jnp.linalg.norm(eigvecs_prev)),
                "eigvals": _trace_array(eigvals[: min(4, eigvals.shape[0])]),
                "eigvecs": _trace_array(eigvecs),
                "state0_overlap_prev": _state0_overlap_prev(grid, backend, eigvecs_prev, eigvecs),
                "occupied_subspace_overlap_prev": _occupied_subspace_overlap_prev(grid, backend, eigvecs_prev, eigvecs, max_m=2),
                "rho_in": _trace_array(rho_cur),
                "rho_new": _trace_array(rho_new),
                "rho_mixed": _trace_array(rho_mixed),
                "rho_update_norm": _rms_norm(rho_new - rho_cur),
                "rho_mix_step_norm": _rms_norm(rho_mixed - rho_cur),
                "rho_new_mixed_diff_norm": _rms_norm(rho_mixed - rho_new),
                "orbital_trace": orbital_trace,
                "eigensolver_stage": stage_name,
                "eigensolver_max_iter_budget": int(orbital_max_iter),
                "eigensolver_tol": float(orbital_tol),
            }
            if orbital_diag is not None:
                payload["eigensolver_inner_iterations"] = int(orbital_diag["iterations"])
                payload["eigensolver_final_residual"] = float(orbital_diag["final_residual"])
                payload["eigensolver_hpsi_calls"] = int(orbital_diag["hpsi_calls"])
            _maybe_trace_append(trace_sink, payload)
        return i + 1, rho_mixed, f_hist_cur, diff, eigvals, eigvecs, V_H, eps_xc, v_xc

    state0 = (i0, rho, f_hist, diff0, eigvals0, eigvecs0, V_H0, eps_xc0, v_xc0)
    if python_control_flow:
        final_state = state0
        while bool(cond(final_state)):
            final_state = body(final_state)
    else:
        final_state = jax.lax.while_loop(cond, body, state0)
        final_state = jax.lax.stop_gradient(final_state)
    i_final, rho, _, diff, eigvals, eigvecs, V_H, eps_xc, v_xc = final_state
    if diagnostics_enabled:
        diagnostics = {
            "result": {
                "final_iterations": int(i_final),
                "final_density_residual": float(diff),
                "converged": bool(float(diff) <= tolerance),
            },
            "scf_history": scf_history,
            "eigensolver_diagnostics": {
                "scheduler_enabled": bool(schedule["enabled"]),
                "density_residual_metric": "max_abs_rho_new_minus_rho_in",
                "stage_switch_threshold": float(schedule["stage_threshold"]) if schedule["enabled"] else None,
                "early_max_iter": None if not schedule["enabled"] else int(schedule["early_max_iter"]),
                "late_max_iter": None if not schedule["enabled"] else int(schedule["late_max_iter"]),
                "late_tol": None if not schedule["enabled"] else float(schedule["late_tol"]),
                "stage_counts": dict(stage_counts),
                "total_inner_iterations": int(eigensolver_total_inner_iterations),
                "total_hpsi_calls": int(eigensolver_total_hpsi_calls),
            },
        }
        return rho, eigvals, eigvecs, V_H, eps_xc, v_xc, diagnostics
    return rho, eigvals, eigvecs, V_H, eps_xc, v_xc


def total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, integration_state, ion_ion, backend=None):
    """Compute the total DFT energy from standard components.

    Args:
        rho: Electron density on the grid, in Bohr^-3.
        eigvals: Kohn-Sham eigenvalues, in Hartree.
        occ: Band occupations (0–2).
        V_loc: Local ionic potential on the grid, in Hartree.
        V_H: Hartree potential on the grid, in Hartree.
        eps_xc: Exchange-correlation energy density on the grid, in Hartree/Bohr^3.
        v_xc: Exchange-correlation potential on the grid, in Hartree.
        volume_element: Grid cell volume, in Bohr^3.
        ion_ion: Ion-ion repulsion energy, in Hartree.

    Returns:
        Total energy in Hartree.
    """
    e_band = jnp.sum(eigvals * occ)
    if backend is None:
        volume_element = integration_state
        e_h_integral = 0.5 * volume_element * jnp.sum(rho * V_H)
        e_xc_integral = volume_element * jnp.sum(eps_xc)
        e_vxc_integral = volume_element * jnp.sum(rho * v_xc)
    else:
        e_h_integral = 0.5 * backend.integrate(integration_state, rho * V_H)
        e_xc_integral = backend.integrate(integration_state, eps_xc)
        e_vxc_integral = backend.integrate(integration_state, rho * v_xc)
    return e_band - e_h_integral + e_xc_integral - e_vxc_integral + ion_ion


def ion_ion_energy(coords, zion):
    """Compute the classical ion-ion Coulomb energy.

    Args:
        coords: Ion coordinates, shape (n_atoms, 3), in Bohr.
        zion: Ionic charges, dimensionless.

    Returns:
        Ion-ion repulsion energy in Hartree.
    """
    e = 0.0
    for i in range(coords.shape[0]):
        for j in range(i + 1, coords.shape[0]):
            r = jnp.linalg.norm(coords[i] - coords[j]) + 1e-12
            e = e + zion[i] * zion[j] / r
    return e


def energy_and_forces(
    grid,
    coords,
    pseudos,
    max_iter,
    mix_alpha,
    tolerance,
    key,
    backend=None,
    *,
    return_diagnostics=False,
    adaptive_scf_aware_eigensolver=None,
    adaptive_eigensolver_early_max_iter=None,
    adaptive_eigensolver_late_max_iter=None,
    adaptive_eigensolver_stage_residual_threshold=None,
    adaptive_eigensolver_late_tol=None,
):
    """Run SCF and return total energy and forces."""
    backend = _resolve_backend(backend)
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=jnp.float32)
    n_electrons = jnp.sum(jnp.asarray([p["q"] for p in pseudos]))
    n_bands = int(jnp.ceil(n_electrons / 2.0))
    occ = jnp.zeros((n_bands,))
    rem = n_electrons
    for i in range(n_bands):
        val = jnp.minimum(2.0, rem)
        occ = occ.at[i].set(val)
        rem -= val
    V_loc = backend.build_local_potential(grid, coords, pseudos)
    scf_result = scf(
        grid,
        coords,
        n_bands,
        occ,
        V_loc,
        pseudos,
        max_iter,
        mix_alpha,
        tolerance,
        key,
        backend=backend,
        return_diagnostics=return_diagnostics,
        adaptive_scf_aware_eigensolver=adaptive_scf_aware_eigensolver,
        adaptive_eigensolver_early_max_iter=adaptive_eigensolver_early_max_iter,
        adaptive_eigensolver_late_max_iter=adaptive_eigensolver_late_max_iter,
        adaptive_eigensolver_stage_residual_threshold=adaptive_eigensolver_stage_residual_threshold,
        adaptive_eigensolver_late_tol=adaptive_eigensolver_late_tol,
    )
    if return_diagnostics:
        rho, eigvals, eigvecs, V_H, eps_xc, v_xc, diagnostics = scf_result
    else:
        rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf_result
        diagnostics = None
    ion_e = ion_ion_energy(coords, zion)
    E_tot = total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid, ion_e, backend=backend)
    forces = jnp.zeros_like(coords)
    if return_diagnostics:
        diagnostics = dict(diagnostics)
        result = dict(diagnostics.get("result", {}))
        result["total_energy"] = float(E_tot)
        diagnostics["result"] = result
        return E_tot, forces, diagnostics
    return E_tot, forces
