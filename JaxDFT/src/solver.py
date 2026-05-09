"""Self-consistent field solver for real-space Kohn-Sham DFT in JAX.

All quantities are in atomic units: length in Bohr, energy in Hartree, and
forces in Hartree/Bohr. The SCF loop builds the effective potential, solves
the Kohn-Sham eigenproblem, and mixes the density until convergence. Force
output is currently a zero placeholder returned by energy_and_forces.
"""

import jax
import jax.numpy as jnp
from .functional import lda_xc
from .hamiltonian import (
    laplacian_4th,
    laplacian_6th,
    laplacian_8th,
    build_local_potential,
    precompute_projectors,
    projector_overlap_diagnostics,
    apply_nonlocal_precomputed,
)


def kinetic_precondition_residuals(residuals, grid_shape, spacing, shift=1.0):
    """Apply a simple FFT kinetic-energy preconditioner to residual vectors."""
    residuals = jnp.asarray(residuals, dtype=jnp.float32)
    n_bands = residuals.shape[1]
    residual_grid = residuals.reshape((*grid_shape, n_bands))
    nx, ny, nz = grid_shape
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=spacing)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=spacing)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=spacing)
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing="ij")
    kinetic = 0.5 * (kx * kx + ky * ky + kz * kz)
    denom = kinetic[..., None] + shift
    residual_k = jnp.fft.fftn(residual_grid, axes=(0, 1, 2))
    preconditioned = jnp.fft.ifftn(residual_k / denom, axes=(0, 1, 2)).real
    return preconditioned.reshape(residuals.shape)


def stabilize_density(rho, volume_element, n_electrons, density_floor=1e-12):
    """Clip mixed density to a nonnegative floor and renormalize charge."""
    dtype = rho.dtype
    rho = jnp.nan_to_num(jnp.asarray(rho, dtype=dtype), nan=density_floor)
    rho = jnp.clip(rho, density_floor, None)
    charge = jnp.sum(rho) * volume_element
    return rho / charge * jnp.asarray(n_electrons, dtype=dtype)


def select_laplacian(order):
    """Return the finite-difference Laplacian implementation for a supported order."""
    if order == 4:
        return laplacian_4th
    if order == 6:
        return laplacian_6th
    if order == 8:
        return laplacian_8th
    raise ValueError("laplacian_order must be 4, 6, or 8")


def solve_orbitals_subspace(
    apply_h_fn,
    n_grid,
    n_bands,
    x_init=None,
    max_iter=100,
    tol=1e-5,
    key=None,
    return_info=False,
    preconditioner_fn=None,
):
    """
    Iterative eigensolver using block subspace expansion + Rayleigh-Ritz.

    Stops when the maximum band residual
        max_i ||H psi_i - eps_i psi_i||_2
    falls below tol.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    dtype = x_init.dtype if x_init is not None else jnp.float32
    if x_init is None:
        X = jax.random.normal(key, (n_grid, n_bands)).astype(dtype)
    else:
        X = x_init + jnp.asarray(1e-6, dtype=dtype) * jax.random.normal(key, (n_grid, n_bands)).astype(dtype)

    # Initial orthonormalization
    X = jnp.linalg.qr(X)[0]
    HX = jax.vmap(apply_h_fn, in_axes=1, out_axes=1)(X)

    def cond_fun(state):
        i, _, _, _, res_norm, _ = state
        return jnp.logical_and(i < max_iter, res_norm > tol)

    def body_fun(state):
        i, X, HX, _, _, _ = state

        # 1) Rayleigh-Ritz in current subspace
        H_sub = X.T @ HX
        H_sub = 0.5 * (H_sub + H_sub.T)   # numerical symmetrization
        E, V_sub = jnp.linalg.eigh(H_sub)
        E = E.astype(dtype)
        V_sub = V_sub.astype(dtype)

        X = (X @ V_sub).astype(dtype)
        HX = (HX @ V_sub).astype(dtype)

        # 2) Residual in current subspace
        R = HX - X * E[None, :]
        res_vec = jnp.sqrt(jnp.sum(R * R, axis=0))
        res_norm = jnp.max(res_vec)

        def done_branch(_):
            return i + 1, X, HX, E, res_norm, res_vec

        def expand_branch(_):
            # 3) Orthogonalize residuals against current subspace
            search = R if preconditioner_fn is None else preconditioner_fn(R)
            R_ortho = search - X @ (X.T @ search)
            R_ortho = jnp.linalg.qr(R_ortho)[0].astype(dtype)

            # 4) Expand subspace
            HR = jax.vmap(apply_h_fn, in_axes=1, out_axes=1)(R_ortho)

            Z = jnp.concatenate([X, R_ortho], axis=1)
            HZ = jnp.concatenate([HX, HR], axis=1)

            # 5) Rayleigh-Ritz in expanded subspace
            H_Z = Z.T @ HZ
            H_Z = 0.5 * (H_Z + H_Z.T)
            E_Z, V_Z = jnp.linalg.eigh(H_Z)
            E_Z = E_Z.astype(dtype)
            V_Z = V_Z.astype(dtype)

            X_new = (Z @ V_Z[:, :n_bands]).astype(dtype)
            HX_new = (HZ @ V_Z[:, :n_bands]).astype(dtype)
            E_new = E_Z[:n_bands]

            # 6) Residual after update
            R_new = HX_new - X_new * E_new[None, :]
            res_vec_new = jnp.sqrt(jnp.sum(R_new * R_new, axis=0))
            res_norm_new = jnp.max(res_vec_new)

            return i + 1, X_new, HX_new, E_new, res_norm_new, res_vec_new

        return jax.lax.cond(res_norm <= tol, done_branch, expand_branch, operand=None)

    state0 = (
        jnp.array(0, dtype=jnp.int32),
        X,
        HX,
        jnp.full((n_bands,), jnp.inf, dtype=dtype),
        jnp.array(jnp.inf, dtype=dtype),
        jnp.full((n_bands,), jnp.inf, dtype=dtype),
    )

    iter_final, X_final, _, E_final, res_final, res_vec_final = jax.lax.while_loop(cond_fun, body_fun, state0)
    if return_info:
        return E_final, X_final, {
            "iterations": iter_final,
            "residual_norm": res_final,
            "residuals": res_vec_final,
        }
    return E_final, X_final


def solve_orbitals_lobpcg(*args, **kwargs):
    """
    Backward-compatible alias.

    The current implementation is a block subspace expansion solver,
    not a strict textbook LOBPCG.
    """
    return solve_orbitals_subspace(*args, **kwargs)


def precompute_poisson_kernel(grid_shape, spacing):
    nx, ny, nz = grid_shape
    x = jnp.fft.fftfreq(2*nx, d=1.0/(2*nx)) * spacing
    y = jnp.fft.fftfreq(2*ny, d=1.0/(2*ny)) * spacing
    z = jnp.fft.fftfreq(2*nz, d=1.0/(2*nz)) * spacing
    KX, KY, KZ = jnp.meshgrid(x, y, z, indexing='ij')
    R = jnp.sqrt(KX**2 + KY**2 + KZ**2)
    
    kernel = jnp.where(R > 1e-8, 1.0 / R, 2.38 / spacing)
    return jnp.fft.fftn(kernel)

@jax.jit
def solve_poisson(rho, kernel_k, spacing):
    nx, ny, nz = rho.shape
    rho_pad = jnp.pad(rho, ((0, nx), (0, ny), (0, nz)), mode='constant')
    
    rho_k = jnp.fft.fftn(rho_pad)
    v_pad = jnp.fft.ifftn(rho_k * kernel_k).real * (spacing**3)
    v = v_pad[:nx, :ny, :nz]
    return v


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


def anderson_mixing(rho, rho_new, f_hist, mix_alpha, iter_idx, m=5, regularization=1e-10):
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
        return (rho + mix_alpha * f).astype(rho.dtype), f_hist0
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
        coeff = jnp.linalg.solve(B + regularization * jnp.eye(m_val), rhs)
        correction = F @ coeff
        rho_next = rho_new - mix_alpha * correction
        rho_next = jnp.nan_to_num(rho_next, nan=rho_new)
        return rho_next.astype(rho.dtype), f_hist1
    return jax.lax.cond(iter_idx == 0, first, later, operand=None)


def pulay_mixing(
    rho,
    rho_new,
    rho_hist,
    f_hist,
    mix_alpha,
    iter_idx,
    m=5,
    regularization=1e-10,
    residual_metric="euclidean",
    grid_shape=None,
    spacing=None,
    kerker_k0=1.0,
):
    """Perform density Pulay/DIIS mixing over damped density candidates."""
    residual = rho_new - rho
    residual = residual - jnp.mean(residual)
    metric_residual = apply_density_residual_metric(
        residual,
        residual_metric,
        grid_shape,
        spacing,
        kerker_k0,
    )
    linear_candidate = rho + mix_alpha * residual
    slot = iter_idx % m
    rho_hist = rho_hist.at[slot].set(linear_candidate)
    f_hist = f_hist.at[slot].set(metric_residual)

    count = jnp.minimum(iter_idx + 1, m)
    active = jnp.arange(m) < count
    b_mat = f_hist @ f_hist.T
    b_mat = b_mat + regularization * jnp.eye(m, dtype=b_mat.dtype)
    active_pair = jnp.logical_and(active[:, None], active[None, :])
    b_mat = jnp.where(active_pair, b_mat, 0.0)
    b_mat = b_mat + jnp.diag(jnp.where(active, 0.0, 1.0))

    aug = jnp.zeros((m + 1, m + 1), dtype=b_mat.dtype)
    aug = aug.at[:m, :m].set(b_mat)
    aug = aug.at[:m, m].set(active.astype(b_mat.dtype))
    aug = aug.at[m, :m].set(active.astype(b_mat.dtype))
    rhs = jnp.zeros((m + 1,), dtype=b_mat.dtype)
    rhs = rhs.at[m].set(1.0)
    coeff = jnp.linalg.solve(aug, rhs)[:m]
    mixed = coeff @ rho_hist
    return mixed, rho_hist, f_hist, coeff


def apply_density_residual_metric(
    residual,
    metric="euclidean",
    grid_shape=None,
    spacing=None,
    kerker_k0=1.0,
):
    """Transform a charge-neutral density residual before Pulay inner products."""
    residual = residual - jnp.mean(residual)
    if metric == "euclidean":
        return residual
    if metric != "kerker":
        raise ValueError("pulay_residual_metric must be 'euclidean' or 'kerker'")
    if grid_shape is None or spacing is None:
        raise ValueError("grid_shape and spacing are required for Kerker residual metric")

    residual_grid = residual.reshape(grid_shape)
    nx, ny, nz = grid_shape
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=spacing)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=spacing)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=spacing)
    kx, ky, kz = jnp.meshgrid(kx, ky, kz, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    k0_sq = jnp.asarray(kerker_k0 * kerker_k0, dtype=jnp.float32)
    weight = k2 / (k2 + k0_sq)
    weight = weight.at[0, 0, 0].set(0.0)
    residual_k = jnp.fft.fftn(residual_grid)
    filtered = jnp.fft.ifftn(residual_k * weight).real
    filtered = filtered - jnp.mean(filtered)
    return filtered.reshape(residual.shape)


def safeguarded_density_mixing(
    rho,
    rho_new,
    anderson_candidate,
    linear_candidate,
    mode="none",
    factor=1.0,
    previous_density_diff=jnp.inf,
    current_density_diff=jnp.inf,
):
    """Optionally reject Anderson steps when the SCF density residual worsens."""
    if mode == "none":
        return anderson_candidate, jnp.array(False)
    if mode != "density_diff":
        raise ValueError("mixing_safeguard must be 'none' or 'density_diff'")
    did_fallback = jnp.logical_and(
        jnp.isfinite(previous_density_diff),
        current_density_diff > factor * previous_density_diff,
    )
    mixed = jnp.where(did_fallback, linear_candidate, anderson_candidate)
    return mixed, did_fallback


def select_scf_residual(metric, density_diff, density_rms_diff, density_l2_diff):
    """Select the scalar residual used by the SCF convergence test."""
    if metric == "max":
        return density_diff
    if metric == "rms":
        return density_rms_diff
    if metric == "l2":
        return density_l2_diff
    raise ValueError("scf_convergence_metric must be 'max', 'rms', or 'l2'")


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
    return_info=False,
    orbital_max_iter=30,
    orbital_tolerance=1e-5,
    orbital_preconditioner="none",
    orbital_preconditioner_shift=1.0,
    mixing_mode="anderson",
    anderson_regularization=1e-10,
    anderson_history=5,
    mixing_safeguard="none",
    mixing_safeguard_factor=1.0,
    scf_convergence_metric="max",
    energy_tolerance=5e-6,
    pulay_residual_metric="euclidean",
    pulay_kerker_k0=1.0,
    laplacian_order=8,
    initial_rho=None,
):
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
        initial_rho: Optional density on ``grid`` (Bohr^-3) to warm-start SCF; clipped
            and renormalized to ``sum(occ)`` electrons. Default uses atom-centered Gaussians.

    Returns:
        Tuple (rho, eigvals, eigvecs, V_H, eps_xc, v_xc) where energies are in
        Hartree and densities in Bohr^-3.
    """
    dtype = grid.coords.dtype
    coords = jnp.asarray(coords, dtype=dtype)
    if key is None:
        key = jax.random.PRNGKey(42)
    if orbital_preconditioner not in ("none", "kinetic"):
        raise ValueError("orbital_preconditioner must be 'none' or 'kinetic'")
    if mixing_mode not in ("anderson", "linear", "pulay"):
        raise ValueError("mixing_mode must be 'anderson', 'linear', or 'pulay'")
    if mixing_safeguard not in ("none", "density_diff"):
        raise ValueError("mixing_safeguard must be 'none' or 'density_diff'")
    if scf_convergence_metric not in ("max", "rms", "l2"):
        raise ValueError("scf_convergence_metric must be 'max', 'rms', or 'l2'")
    if pulay_residual_metric not in ("euclidean", "kerker"):
        raise ValueError("pulay_residual_metric must be 'euclidean' or 'kerker'")
    laplacian_fn = select_laplacian(laplacian_order)
    if anderson_history < 1:
        raise ValueError("anderson_history must be >= 1")
    volume_element = grid.volume_element

    if initial_rho is None:
        rho = jnp.zeros(grid.shape, dtype=dtype)
        for a in range(coords.shape[0]):
            r = jnp.linalg.norm(grid.coords - coords[a], axis=-1)
            rho = rho + jnp.exp(-2.0 * r**2)
        rho = rho / (jnp.sum(rho) * volume_element) * jnp.sum(occ)
    else:
        rho = jnp.asarray(initial_rho, dtype=dtype).reshape(grid.shape)
        if rho.shape != grid.shape:
            raise ValueError(f"initial_rho shape {rho.shape} must match grid.shape {grid.shape}")
        rho = stabilize_density(rho, volume_element, jnp.sum(occ))

    f_hist = jnp.zeros((anderson_history, rho.size), dtype=dtype)
    rho_hist = jnp.zeros((anderson_history, rho.size), dtype=dtype)
    n_grid = rho.size
    kernel_k = precompute_poisson_kernel(grid.shape, grid.spacing)
    proj_data = precompute_projectors(grid, coords, projectors)
    preconditioner_fn = None
    if orbital_preconditioner == "kinetic":
        preconditioner_fn = lambda residuals: kinetic_precondition_residuals(
            residuals,
            grid.shape,
            grid.spacing,
            orbital_preconditioner_shift,
        )
    # 占位符
    eigvals0 = jnp.zeros((n_bands,), dtype=dtype)
    eigvecs0 = jnp.zeros((n_grid, n_bands), dtype=dtype)
    V_H0 = jnp.zeros(grid.shape, dtype=dtype)
    eps_xc0 = jnp.zeros(grid.shape, dtype=dtype)
    v_xc0 = jnp.zeros(grid.shape, dtype=dtype)
    diff0 = jnp.array(jnp.inf, dtype=dtype)
    orbital_res0 = jnp.array(jnp.inf, dtype=dtype)
    orbital_iter0 = jnp.array(0, dtype=jnp.int32)
    orbital_res_vec0 = jnp.full((n_bands,), jnp.inf, dtype=dtype)
    energy_history0 = jnp.full((max_iter,), jnp.nan, dtype=dtype)
    energy_delta_history0 = jnp.full((max_iter,), jnp.nan, dtype=dtype)
    density_diff_history0 = jnp.full((max_iter,), jnp.nan, dtype=dtype)
    density_rms_history0 = jnp.full((max_iter,), jnp.nan, dtype=dtype)
    density_l2_history0 = jnp.full((max_iter,), jnp.nan, dtype=dtype)
    orbital_residual_history0 = jnp.full((max_iter,), jnp.nan, dtype=dtype)
    fallback_count0 = jnp.array(0, dtype=jnp.int32)
    convergence_residual0 = jnp.array(jnp.inf, dtype=dtype)
    i0 = jnp.array(0, dtype=jnp.int32)

    def cond(state):
        i = state[0]
        convergence_residual = state[-1]
        return jnp.logical_and(i < max_iter, convergence_residual > tolerance)

    def body(state):
        (
            i,
            rho_cur,
            f_hist_cur,
            rho_hist_cur,
            previous_diff,
            _,
            eigvecs0,
            V_H_prev,
            _,
            _,
            _,
            _,
            _,
            energy_history,
            energy_delta_history,
            density_diff_history,
            density_rms_history,
            density_l2_history,
            orbital_residual_history,
            fallback_count,
            _,
        ) = state
        rho_cur = jnp.clip(rho_cur, 1e-12, None)
        

        V_H = solve_poisson(rho_cur, kernel_k, grid.spacing).astype(dtype)
        
        eps_xc, v_xc = lda_xc(rho_cur)
        eps_xc = eps_xc.astype(dtype)
        v_xc = v_xc.astype(dtype)
        V_eff = V_loc + V_H + v_xc

        def apply_h(psi_flat):
            psi = psi_flat.reshape(grid.shape)
            lap = laplacian_fn(psi, grid.spacing, grid.mask)
            
            if proj_data is not None:
                P_i, P_j, coeffs = proj_data
                v_nonlocal = apply_nonlocal_precomputed(psi, P_i, P_j, coeffs, grid.volume_element)
            else:
                v_nonlocal = jnp.zeros_like(psi)
            
            hpsi = -0.5 * lap + V_eff * psi + v_nonlocal # 加上非局域项
            return hpsi.reshape(-1).astype(dtype)

        # Subspace eigensolver with residual-based convergence
        iter_key = jax.random.fold_in(key, i)
        eigvals, eigvecs, orbital_info = solve_orbitals_subspace(
            apply_h,
            n_grid,
            n_bands,
            x_init=eigvecs0,
            max_iter=orbital_max_iter,
            tol=orbital_tolerance,
            key=iter_key,
            return_info=True,
            preconditioner_fn=preconditioner_fn,
        )
        
        # 归一化
        norm = jnp.sqrt(jnp.sum(eigvecs**2, axis=0) * volume_element)
        eigvecs = eigvecs / norm
        
        rho_new = jnp.sum((eigvecs ** 2) * occ[None, :], axis=1).reshape(grid.shape)
        density_residual = rho_new - rho_cur
        diff = jnp.max(jnp.abs(density_residual))
        density_rms = jnp.sqrt(jnp.mean(density_residual * density_residual))
        density_l2 = jnp.sqrt(grid.volume_element * jnp.sum(density_residual * density_residual))
        convergence_residual = select_scf_residual(
            scf_convergence_metric,
            diff,
            density_rms,
            density_l2,
        )
        energy_iter = total_energy(
            rho_cur,
            eigvals,
            occ,
            V_loc,
            V_H,
            eps_xc,
            v_xc,
            grid.volume_element,
            0.0,
        ).astype(dtype)
        previous_energy = energy_history[jnp.maximum(i - 1, 0)]
        energy_delta = jax.lax.cond(
            i == 0,
            lambda _: jnp.array(jnp.inf, dtype=dtype),
            lambda _: jnp.abs(energy_iter - previous_energy),
            operand=None,
        )
        energy_history = energy_history.at[i].set(energy_iter)
        energy_delta_history = energy_delta_history.at[i].set(energy_delta)
        density_diff_history = density_diff_history.at[i].set(diff)
        density_rms_history = density_rms_history.at[i].set(density_rms)
        density_l2_history = density_l2_history.at[i].set(density_l2)
        orbital_residual_history = orbital_residual_history.at[i].set(orbital_info["residual_norm"])
        
        linear_flat = rho_cur.reshape(-1) + mix_alpha * (rho_new.reshape(-1) - rho_cur.reshape(-1))
        if mixing_mode == "linear":
            rho_flat = linear_flat
            did_fallback = jnp.array(False)
            rho_hist_next = rho_hist_cur
        else:
            if mixing_mode == "pulay":
                mixed_candidate, rho_hist_next, f_hist_cur, _ = pulay_mixing(
                    rho_cur.reshape(-1),
                    rho_new.reshape(-1),
                    rho_hist_cur,
                    f_hist_cur,
                    mix_alpha,
                    i,
                    m=anderson_history,
                    regularization=anderson_regularization,
                    residual_metric=pulay_residual_metric,
                    grid_shape=grid.shape,
                    spacing=grid.spacing,
                    kerker_k0=pulay_kerker_k0,
                )
            else:
                mixed_candidate, f_hist_cur = anderson_mixing(
                    rho_cur.reshape(-1),
                    rho_new.reshape(-1),
                    f_hist_cur,
                    mix_alpha,
                    i,
                    m=anderson_history,
                    regularization=anderson_regularization,
                )
                rho_hist_next = rho_hist_cur
            rho_flat, did_fallback = safeguarded_density_mixing(
            rho_cur.reshape(-1),
            rho_new.reshape(-1),
            mixed_candidate,
            linear_flat,
            mode=mixing_safeguard,
            factor=mixing_safeguard_factor,
            previous_density_diff=previous_diff,
            current_density_diff=diff,
        )
        fallback_count = fallback_count + did_fallback.astype(jnp.int32)
        rho_next = stabilize_density(
            rho_flat.reshape(grid.shape),
            grid.volume_element,
            jnp.sum(occ),
        )
        return (
            i + 1,
            rho_next,
            f_hist_cur,
            rho_hist_next,
            diff,
            eigvals,
            eigvecs,
            V_H,
            eps_xc,
            v_xc,
            orbital_info["residual_norm"],
            orbital_info["iterations"],
            orbital_info["residuals"],
            energy_history,
            energy_delta_history,
            density_diff_history,
            density_rms_history,
            density_l2_history,
            orbital_residual_history,
            fallback_count,
            convergence_residual,
        )

    state0 = (
        i0,
        rho,
        f_hist,
        rho_hist,
        diff0,
        eigvals0,
        eigvecs0,
        V_H0,
        eps_xc0,
        v_xc0,
        orbital_res0,
        orbital_iter0,
        orbital_res_vec0,
        energy_history0,
        energy_delta_history0,
        density_diff_history0,
        density_rms_history0,
        density_l2_history0,
        orbital_residual_history0,
        fallback_count0,
        convergence_residual0,
    )
    final_state = jax.lax.while_loop(cond, body, state0)
    
    # 停止梯度并取回结果
    final_state = jax.lax.stop_gradient(final_state)
    (
        scf_iter,
        rho,
        _,
        _,
        diff,
        eigvals,
        eigvecs,
        V_H,
        eps_xc,
        v_xc,
        orbital_res,
        orbital_iter,
        orbital_res_vec,
        energy_history,
        energy_delta_history,
        density_diff_history,
        density_rms_history,
        density_l2_history,
        orbital_residual_history,
        fallback_count,
        convergence_residual,
    ) = final_state

    last_idx = jnp.maximum(scf_iter - 1, 0)
    history_idx = jnp.arange(max_iter)
    last10_start = jnp.maximum(scf_iter - 10, 0)
    last10_mask = jnp.logical_and(history_idx >= last10_start, history_idx < scf_iter)
    energy_delta_last = energy_delta_history[last_idx]
    energy_delta_last10_max = jnp.max(jnp.where(last10_mask, energy_delta_history, -jnp.inf))
    density_converged = convergence_residual <= tolerance
    energy_converged = energy_delta_last10_max <= energy_tolerance

    if return_info:
        return rho, eigvals, eigvecs, V_H, eps_xc, v_xc, {
            "iterations": scf_iter,
            "density_diff": diff,
            "density_rms_diff": density_rms_history[last_idx],
            "density_l2_diff": density_l2_history[last_idx],
            "scf_convergence_residual": convergence_residual,
            "scf_convergence_metric": scf_convergence_metric,
            "energy_delta_last": energy_delta_last,
            "energy_delta_history": energy_delta_history,
            "energy_delta_last10_max": energy_delta_last10_max,
            "density_converged": density_converged,
            "energy_converged": energy_converged,
            "orbital_residual": orbital_res,
            "orbital_residuals": orbital_res_vec,
            "orbital_iterations": orbital_iter,
            "scf_converged": jnp.logical_and(density_converged, energy_converged),
            "orbital_converged": orbital_res <= orbital_tolerance,
            "orbital_max_iter": orbital_max_iter,
            "orbital_tolerance": orbital_tolerance,
            "orbital_preconditioner": orbital_preconditioner,
            "orbital_preconditioner_shift": orbital_preconditioner_shift,
            "mixing_mode": mixing_mode,
            "anderson_regularization": anderson_regularization,
            "anderson_history": anderson_history,
            "mixing_safeguard": mixing_safeguard,
            "mixing_safeguard_factor": mixing_safeguard_factor,
            "mixing_fallback_count": fallback_count,
            "pulay_residual_metric": pulay_residual_metric,
            "pulay_kerker_k0": pulay_kerker_k0,
            "laplacian_order": laplacian_order,
            "energy_history": energy_history,
            "density_diff_history": density_diff_history,
            "density_rms_diff_history": density_rms_history,
            "density_l2_diff_history": density_l2_history,
            "orbital_residual_history": orbital_residual_history,
            "density_min": jnp.min(rho),
            "density_integral": jnp.sum(rho) * grid.volume_element,
        }
    return rho, eigvals, eigvecs, V_H, eps_xc, v_xc


def total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, volume_element, ion_ion):
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
    e_h_integral = 0.5 * volume_element * jnp.sum(rho * V_H)
    e_xc_integral = volume_element * jnp.sum(eps_xc)
    e_vxc_integral = volume_element * jnp.sum(rho * v_xc)
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


def nonlocal_pseudopotential_energy(eigvecs, occ, grid_shape, projectors, volume_element):
    """Compute sum_n f_n <psi_n|V_nonlocal|psi_n> for diagnostics."""
    if projectors is None:
        return jnp.asarray(0.0, dtype=eigvecs.dtype)
    p_i, p_j, coeffs = projectors
    energy = jnp.asarray(0.0, dtype=eigvecs.dtype)
    for band in range(eigvecs.shape[1]):
        psi = eigvecs[:, band].reshape(grid_shape)
        vnl_psi = apply_nonlocal_precomputed(psi, p_i, p_j, coeffs, volume_element)
        energy = energy + occ[band] * jnp.sum(psi * vnl_psi) * volume_element
    return energy


def local_pseudopotential_energy_by_atom(rho, grid, coords, zion, rloc, c):
    """Compute per-atom local pseudopotential energy contributions."""
    energies = []
    for atom_idx in range(coords.shape[0]):
        V_atom = build_local_potential(
            coords[atom_idx : atom_idx + 1],
            grid.coords,
            zion[atom_idx : atom_idx + 1],
            rloc[atom_idx : atom_idx + 1],
            c[atom_idx : atom_idx + 1],
        )
        energies.append(grid.volume_element * jnp.sum(rho * V_atom))
    if not energies:
        return jnp.zeros((0,), dtype=grid.coords.dtype)
    return jnp.stack(energies)


def local_pseudopotential_integral_by_atom(grid, coords, zion, rloc, c):
    """Compute grid integrals of each atom's local GTH potential."""
    integrals = []
    for atom_idx in range(coords.shape[0]):
        V_atom = build_local_potential(
            coords[atom_idx : atom_idx + 1],
            grid.coords,
            zion[atom_idx : atom_idx + 1],
            rloc[atom_idx : atom_idx + 1],
            c[atom_idx : atom_idx + 1],
        )
        integrals.append(grid.volume_element * jnp.sum(V_atom))
    if not integrals:
        return jnp.zeros((0,), dtype=grid.coords.dtype)
    return jnp.stack(integrals)


def energy_and_forces(
    grid,
    coords,
    pseudos,
    max_iter,
    mix_alpha,
    tolerance,
    key,
    return_info=False,
    orbital_max_iter=30,
    orbital_tolerance=1e-5,
    orbital_preconditioner="none",
    orbital_preconditioner_shift=1.0,
    mixing_mode="anderson",
    anderson_regularization=1e-10,
    anderson_history=5,
    mixing_safeguard="none",
    mixing_safeguard_factor=1.0,
    scf_convergence_metric="max",
    energy_tolerance=5e-6,
    pulay_residual_metric="euclidean",
    pulay_kerker_k0=1.0,
    laplacian_order=8,
    initial_rho=None,
):
    """Run SCF and return total energy and forces.

    Args:
        grid: Grid object produced by create_grid.
        coords: Ion coordinates, shape (n_atoms, 3), in Bohr.
        pseudos: List of pseudopotential dictionaries.
        max_iter: Maximum SCF iterations.
        mix_alpha: Anderson mixing strength.
        tolerance: Convergence threshold for density change.
        key: Base JAX PRNG key used to seed the SCF orbital initialization.
        initial_rho: Optional warm-start density on ``grid`` (see ``scf``).

    Returns:
        Tuple (energy, forces) where energy is in Hartree and forces are in
        Hartree/Bohr. Forces are currently zeros in this implementation.
    """
    dtype = grid.coords.dtype
    zion = jnp.asarray([p["zion"] for p in pseudos], dtype=dtype)
    rloc = jnp.asarray([p["rloc"] for p in pseudos], dtype=dtype)
    c = jnp.asarray([p["c"] for p in pseudos], dtype=dtype)
    
    n_electrons = jnp.sum(jnp.asarray([p["q"] for p in pseudos], dtype=dtype))
    n_bands = int(jnp.ceil(n_electrons / 2.0))
    occ = jnp.zeros((n_bands,), dtype=dtype)
    rem = n_electrons
    for i in range(n_bands):
        val = jnp.minimum(2.0, rem)
        occ = occ.at[i].set(val)
        rem -= val

    V_loc = build_local_potential(coords, grid.coords, zion, rloc, c)
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
        return_info=return_info,
        orbital_max_iter=orbital_max_iter,
        orbital_tolerance=orbital_tolerance,
        orbital_preconditioner=orbital_preconditioner,
        orbital_preconditioner_shift=orbital_preconditioner_shift,
        mixing_mode=mixing_mode,
        anderson_regularization=anderson_regularization,
        anderson_history=anderson_history,
        mixing_safeguard=mixing_safeguard,
        mixing_safeguard_factor=mixing_safeguard_factor,
        scf_convergence_metric=scf_convergence_metric,
        energy_tolerance=energy_tolerance,
        pulay_residual_metric=pulay_residual_metric,
        pulay_kerker_k0=pulay_kerker_k0,
        laplacian_order=laplacian_order,
        initial_rho=initial_rho,
    )
    if return_info:
        rho, eigvals, eigvecs, V_H, eps_xc, v_xc, scf_info = scf_result
    else:
        rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf_result
    
    ion_e = ion_ion_energy(coords, zion)
    E_tot = total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid.volume_element, ion_e)
    forces = jnp.zeros_like(coords)
    if return_info:
        proj_data = precompute_projectors(grid, coords, pseudos)
        local_energy = grid.volume_element * jnp.sum(rho * V_loc)
        local_energy_by_atom = local_pseudopotential_energy_by_atom(
            rho,
            grid,
            coords,
            zion,
            rloc,
            c,
        )
        local_integral_by_atom = local_pseudopotential_integral_by_atom(
            grid,
            coords,
            zion,
            rloc,
            c,
        )
        nonlocal_energy = nonlocal_pseudopotential_energy(
            eigvecs,
            occ,
            grid.shape,
            proj_data,
            grid.volume_element,
        )
        energy_components = {
            "band": jnp.sum(eigvals * occ),
            "local_pseudopotential": local_energy,
            "local_pseudopotential_by_atom": local_energy_by_atom,
            "nonlocal_pseudopotential": nonlocal_energy,
            "hartree": 0.5 * grid.volume_element * jnp.sum(rho * V_H),
            "xc": grid.volume_element * jnp.sum(eps_xc),
            "vxc": grid.volume_element * jnp.sum(rho * v_xc),
            "ion_ion": ion_e,
        }
        info = {
            "scf_iterations": scf_info["iterations"],
            "density_diff": scf_info["density_diff"],
            "density_rms_diff": scf_info["density_rms_diff"],
            "density_l2_diff": scf_info["density_l2_diff"],
            "scf_convergence_residual": scf_info["scf_convergence_residual"],
            "scf_convergence_metric": scf_info["scf_convergence_metric"],
            "energy_delta_last": scf_info["energy_delta_last"],
            "energy_delta_history": scf_info["energy_delta_history"],
            "energy_delta_last10_max": scf_info["energy_delta_last10_max"],
            "density_converged": scf_info["density_converged"],
            "energy_converged": scf_info["energy_converged"],
            "orbital_residual": scf_info["orbital_residual"],
            "orbital_residuals": scf_info["orbital_residuals"],
            "energy_components": energy_components,
            "n_bands": n_bands,
            "occupations": occ,
            "eigenvalues": eigvals,
            "scf_converged": scf_info["scf_converged"],
            "orbital_iterations": scf_info["orbital_iterations"],
            "orbital_converged": scf_info["orbital_converged"],
            "orbital_max_iter": scf_info["orbital_max_iter"],
            "orbital_tolerance": scf_info["orbital_tolerance"],
            "orbital_preconditioner": scf_info["orbital_preconditioner"],
            "orbital_preconditioner_shift": scf_info["orbital_preconditioner_shift"],
            "mixing_mode": scf_info["mixing_mode"],
            "anderson_regularization": scf_info["anderson_regularization"],
            "anderson_history": scf_info["anderson_history"],
            "mixing_safeguard": scf_info["mixing_safeguard"],
            "mixing_safeguard_factor": scf_info["mixing_safeguard_factor"],
            "mixing_fallback_count": scf_info["mixing_fallback_count"],
            "pulay_residual_metric": scf_info["pulay_residual_metric"],
            "pulay_kerker_k0": scf_info["pulay_kerker_k0"],
            "laplacian_order": scf_info["laplacian_order"],
            "energy_history": scf_info["energy_history"] + ion_e,
            "local_pseudopotential_min": jnp.min(V_loc),
            "local_pseudopotential_max": jnp.max(V_loc),
            "local_pseudopotential_integral": grid.volume_element * jnp.sum(V_loc),
            "local_pseudopotential_integral_by_atom": local_integral_by_atom,
            "hartree_potential_min": jnp.min(V_H),
            "hartree_potential_max": jnp.max(V_H),
            "hartree_potential_integral": grid.volume_element * jnp.sum(V_H),
            "density_diff_history": scf_info["density_diff_history"],
            "density_rms_diff_history": scf_info["density_rms_diff_history"],
            "density_l2_diff_history": scf_info["density_l2_diff_history"],
            "orbital_residual_history": scf_info["orbital_residual_history"],
            "projector_overlap_diagnostics": projector_overlap_diagnostics(grid, coords, pseudos),
            "density_min": scf_info["density_min"],
            "density_integral": scf_info["density_integral"],
            "density": rho,
        }
        return E_tot, forces, info
    return E_tot, forces
