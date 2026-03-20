"""Self-consistent field solver for real-space Kohn-Sham DFT in JAX.

All quantities are in atomic units: length in Bohr, energy in Hartree, and
forces in Hartree/Bohr. The SCF loop builds the effective potential, solves
the Kohn-Sham eigenproblem, and mixes the density until convergence.
"""

import jax
import jax.numpy as jnp
from .backends.uniform import (
    UniformBackend,
    precompute_uniform_poisson_kernel,
    solve_uniform_poisson,
)
from .functional import lda_xc


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


def _metric_orthonormalize(grid, backend, X, scale, eps=1e-8):
    """Metric-orthonormalize a block of vectors using its small Gram matrix."""
    S = _metric_symmetrize(_metric_gram(grid, backend, X, X, scale))
    evals, evecs = jnp.linalg.eigh(S)
    eps = jnp.asarray(eps, dtype=evals.dtype)
    safe_evals = jnp.maximum(evals, eps)
    inv_sqrt = 1.0 / jnp.sqrt(safe_evals)
    return X @ (evecs * inv_sqrt[None, :])


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
    X = _metric_orthonormalize(grid, backend, X, scale)
    HX = jax.vmap(apply_h_fn, in_axes=1, out_axes=1)(X)
    E = jnp.full((n_bands,), jnp.inf, dtype=jnp.float32)
    res_norm = jnp.array(jnp.inf, dtype=jnp.float32)
    i = 0

    while i < max_iter and float(res_norm) > tol:
        # 1) Rayleigh-Ritz in current metric-orthonormal subspace
        H_sub = _metric_symmetrize(_metric_gram(grid, backend, X, HX, scale))
        E, V_sub = jnp.linalg.eigh(H_sub)

        X = X @ V_sub
        HX = HX @ V_sub

        # 2) Residual in current subspace
        R = HX - X * E[None, :]
        res_vec = _metric_block_norms(grid, backend, R, scale)
        res_norm = jnp.max(res_vec)
        if float(res_norm) <= tol:
            i += 1
            break

        # 3) Orthogonalize residuals against current subspace in metric W
        R_ortho = _metric_project_out(grid, backend, X, R, scale)
        R_ortho = _metric_orthonormalize(grid, backend, R_ortho, scale)

        # 4) Expand and re-orthonormalize the full subspace for stability
        Z = jnp.concatenate([X, R_ortho], axis=1)
        Z = _metric_orthonormalize(grid, backend, Z, scale)
        HZ = jax.vmap(apply_h_fn, in_axes=1, out_axes=1)(Z)

        # 5) Rayleigh-Ritz in expanded metric-orthonormal subspace
        H_Z = _metric_symmetrize(_metric_gram(grid, backend, Z, HZ, scale))
        E_Z, V_Z = jnp.linalg.eigh(H_Z)

        X = Z @ V_Z[:, :n_bands]
        HX = HZ @ V_Z[:, :n_bands]
        E = E_Z[:n_bands]

        # 6) Residual after update
        R_new = HX - X * E[None, :]
        res_vec_new = _metric_block_norms(grid, backend, R_new, scale)
        res_norm = jnp.max(res_vec_new)
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


def scf(grid, coords, n_bands, occ, V_loc, projectors, max_iter, mix_alpha, tolerance, key, backend=None):
    """Run the self-consistent field (SCF) loop.

    Args:
        grid: Grid object with coordinates, spacing, and volume element.
        coords: Ion coordinates, shape (n_atoms, 3), in Bohr.
        n_bands: Number of Kohn-Sham orbitals to solve for.
        occ: Band occupations (0–2).
        V_loc: Local ionic potential on the grid, in Hartree.
        projectors: Nonlocal projector data structure.
        max_iter: Maximum SCF iterations.
        mix_alpha: Anderson mixing strength.
        tolerance: Convergence threshold for density change.
        key: Base JAX PRNG key used to seed orbital initialization.

    Returns:
        Tuple (rho, eigvals, eigvecs, V_H, eps_xc, v_xc) where energies are in
        Hartree and densities in Bohr^-3.
    """
    backend = _resolve_backend(backend)
    coords = jnp.asarray(coords, dtype=jnp.float32)
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # 初始密度
    rho = jnp.zeros(grid.shape, dtype=jnp.float32)
    for a in range(coords.shape[0]):
        r = jnp.linalg.norm(grid.coords - coords[a], axis=-1)
        rho = rho + jnp.exp(-2.0 * r**2)
    rho = rho / backend.integrate(grid, rho) * jnp.sum(occ)

    f_hist = jnp.zeros((5, rho.size), dtype=jnp.float32)
    n_grid = rho.size
    proj_data = backend.precompute_nonlocal(grid, coords, projectors)
    # 占位符
    eigvals0 = jnp.zeros((n_bands,), dtype=jnp.float32)
    eigvecs0 = jnp.zeros((n_grid, n_bands), dtype=jnp.float32)
    V_H0 = jnp.zeros(grid.shape, dtype=jnp.float32)
    eps_xc0 = jnp.zeros(grid.shape, dtype=jnp.float32)
    v_xc0 = jnp.zeros(grid.shape, dtype=jnp.float32)
    diff0 = jnp.array(jnp.inf, dtype=jnp.float32)
    i0 = jnp.array(0, dtype=jnp.int32)

    def cond(state):
        i, _, _, diff, _, _, _, _, _ = state
        return jnp.logical_and(i < max_iter, diff > tolerance)

    def body(state):
        i, rho_cur, f_hist_cur, diff, _, eigvecs0, V_H_prev, _, _ = state
        rho_cur = jnp.clip(rho_cur, 1e-12, None)
        

        V_H = backend.solve_hartree(grid, rho_cur)
        
        eps_xc, v_xc = lda_xc(rho_cur)
        V_eff = V_loc + V_H + v_xc

        def apply_h(psi_flat):
            psi = psi_flat.reshape(grid.shape)
            kinetic_psi = backend.apply_kinetic(grid, psi)
            v_nonlocal = backend.apply_nonlocal(grid, psi, proj_data)

            hpsi = kinetic_psi + V_eff * psi + v_nonlocal # ?????????
            return hpsi.reshape(-1)

        # Subspace eigensolver with residual-based convergence
        iter_key = jax.random.fold_in(key, i)
        metric_backend = backend if getattr(backend, "name", None) != "uniform" else None
        metric_grid = grid if metric_backend is not None else None
        orbital_max_iter = 30 if metric_backend is None else 8
        orbital_tol = 1e-5 if metric_backend is None else 1e-4
        eigvals, eigvecs = solve_orbitals_subspace(
            apply_h,
            n_grid,
            n_bands,
            x_init=eigvecs0,
            max_iter=orbital_max_iter,
            tol=orbital_tol,
            key=iter_key,
            grid=metric_grid,
            backend=metric_backend,
        )
        
        # 归一化
        eigvec_fields = jnp.moveaxis(eigvecs.reshape(grid.shape + (n_bands,)), -1, 0)
        norm = jnp.sqrt(jax.vmap(lambda psi: backend.inner_product(grid, psi, psi))(eigvec_fields))
        eigvecs = eigvecs / norm[None, :]
        
        rho_new = jnp.sum((eigvecs ** 2) * occ[None, :], axis=1).reshape(grid.shape)
        diff = jnp.max(jnp.abs(rho_new - rho_cur))
        
        rho_flat, f_hist_cur = anderson_mixing(
            rho_cur.reshape(-1), rho_new.reshape(-1), f_hist_cur, mix_alpha, i
        )
        return i + 1, rho_flat.reshape(grid.shape), f_hist_cur, diff, eigvals, eigvecs, V_H, eps_xc, v_xc

    state0 = (i0, rho, f_hist, diff0, eigvals0, eigvecs0, V_H0, eps_xc0, v_xc0)
    if getattr(backend, "name", None) == "uniform":
        final_state = jax.lax.while_loop(cond, body, state0)
        final_state = jax.lax.stop_gradient(final_state)
    else:
        final_state = state0
        while bool(cond(final_state)):
            final_state = body(final_state)
    _, rho, _, diff, eigvals, eigvecs, V_H, eps_xc, v_xc = final_state
    
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


def energy_and_forces(grid, coords, pseudos, max_iter, mix_alpha, tolerance, key, backend=None):
    """Run SCF and return total energy and forces.

    Args:
        grid: Grid object produced by create_grid.
        coords: Ion coordinates, shape (n_atoms, 3), in Bohr.
        pseudos: List of pseudopotential dictionaries.
        max_iter: Maximum SCF iterations.
        mix_alpha: Anderson mixing strength.
        tolerance: Convergence threshold for density change.
        key: Base JAX PRNG key used to seed the SCF orbital initialization.

    Returns:
        Tuple (energy, forces) where energy is in Hartree and forces are in
        Hartree/Bohr. Forces are currently zeros in this implementation.
    """
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
    rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
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
    )
    
    ion_e = ion_ion_energy(coords, zion)
    E_tot = total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid, ion_e, backend=backend)
    return E_tot, jnp.zeros_like(coords)
