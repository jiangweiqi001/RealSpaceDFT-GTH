"""Self-consistent field solver for real-space Kohn-Sham DFT in JAX.

All quantities are in atomic units: length in Bohr, energy in Hartree, and
forces in Hartree/Bohr. The SCF loop builds the effective potential, solves
the Kohn-Sham eigenproblem, and mixes the density until convergence.
"""

import jax
import jax.numpy as jnp
from .functional import lda_xc
from .hamiltonian import (
    laplacian_8th,
    build_local_potential,
    precompute_projectors,
    apply_nonlocal_precomputed,
    apply_nonlocal_fine_integral,
)


def solve_orbitals_subspace(
    apply_h_fn,
    n_grid,
    n_bands,
    x_init=None,
    max_iter=100,
    tol=1e-5,
    key=None,
):
    """
    Iterative eigensolver using block subspace expansion + Rayleigh-Ritz.

    Stops when the maximum band residual
        max_i ||H psi_i - eps_i psi_i||_2
    falls below tol.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    if x_init is None:
        X = jax.random.normal(key, (n_grid, n_bands)).astype(jnp.float32)
    else:
        X = x_init + 1e-6 * jax.random.normal(key, (n_grid, n_bands)).astype(jnp.float32)

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
    projector_subgrid=1,
    projector_mode="cell_average",
    projector_patch_radius_factor=6.0,
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
        projector_subgrid: Number of cell-average samples per axis for
            nonlocal projectors. The default 1 keeps point sampling.
        projector_mode: "cell_average" for coarse-cell averaging or "patch"
            for atom-centered fine-grid projector integration.
        projector_patch_radius_factor: Patch radius multiplier applied to each
            projector radius in patch mode.

    Returns:
        Tuple (rho, eigvals, eigvecs, V_H, eps_xc, v_xc) where energies are in
        Hartree and densities in Bohr^-3.
    """
    coords = jnp.asarray(coords, dtype=jnp.float32)
    if key is None:
        key = jax.random.PRNGKey(42)
    volume_element = grid.volume_element
    
    # 初始密度
    rho = jnp.zeros(grid.shape, dtype=jnp.float32)
    for a in range(coords.shape[0]):
        r = jnp.linalg.norm(grid.coords - coords[a], axis=-1)
        rho = rho + jnp.exp(-2.0 * r**2)
    rho = rho / (jnp.sum(rho) * volume_element) * jnp.sum(occ)

    f_hist = jnp.zeros((5, rho.size), dtype=jnp.float32)
    n_grid = rho.size
    kernel_k = precompute_poisson_kernel(grid.shape, grid.spacing)
    proj_data = precompute_projectors(
        grid,
        coords,
        projectors,
        projector_subgrid=projector_subgrid,
        projector_mode=projector_mode,
        projector_patch_radius_factor=projector_patch_radius_factor,
    )
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
        

        V_H = solve_poisson(rho_cur, kernel_k, grid.spacing)
        
        eps_xc, v_xc = lda_xc(rho_cur)
        V_eff = V_loc + V_H + v_xc

        def apply_h(psi_flat):
            psi = psi_flat.reshape(grid.shape)
            lap = laplacian_8th(psi, grid.spacing, grid.mask)
            
            if proj_data is not None:
                if proj_data[0] == "fine_integral":
                    v_nonlocal = apply_nonlocal_fine_integral(psi, *proj_data[1:])
                else:
                    P_i, P_j, coeffs = proj_data
                    v_nonlocal = apply_nonlocal_precomputed(psi, P_i, P_j, coeffs, grid.volume_element)
            else:
                v_nonlocal = jnp.zeros_like(psi)
            
            hpsi = -0.5 * lap + V_eff * psi + v_nonlocal # 加上非局域项
            return hpsi.reshape(-1)

        # Subspace eigensolver with residual-based convergence
        iter_key = jax.random.fold_in(key, i)
        eigvals, eigvecs = solve_orbitals_subspace(
    apply_h,
    n_grid,
    n_bands,
    x_init=eigvecs0,
    max_iter=30,
    tol=1e-5,
    key=iter_key,
)
        
        # 归一化
        norm = jnp.sqrt(jnp.sum(eigvecs**2, axis=0) * volume_element)
        eigvecs = eigvecs / norm
        
        rho_new = jnp.sum((eigvecs ** 2) * occ[None, :], axis=1).reshape(grid.shape)
        diff = jnp.max(jnp.abs(rho_new - rho_cur))
        
        rho_flat, f_hist_cur = anderson_mixing(
            rho_cur.reshape(-1), rho_new.reshape(-1), f_hist_cur, mix_alpha, i
        )
        return i + 1, rho_flat.reshape(grid.shape), f_hist_cur, diff, eigvals, eigvecs, V_H, eps_xc, v_xc

    state0 = (i0, rho, f_hist, diff0, eigvals0, eigvecs0, V_H0, eps_xc0, v_xc0)
    final_state = jax.lax.while_loop(cond, body, state0)
    
    # 停止梯度并取回结果
    final_state = jax.lax.stop_gradient(final_state)
    _, rho, _, diff, eigvals, eigvecs, V_H, eps_xc, v_xc = final_state
    
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


def energy_and_forces(
    grid,
    coords,
    pseudos,
    max_iter,
    mix_alpha,
    tolerance,
    key,
    fine_grid_mode=None,
    fine_subgrid=5,
    fine_grid_radius_factor=4.0,
    local_subgrid=1,
    local_mode="cell_average",
    local_patch_radius_factor=6.0,
    projector_subgrid=1,
    projector_mode="cell_average",
    projector_patch_radius_factor=6.0,
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
        fine_grid_mode: Unified fine-grid mode. None keeps the explicit
            local/projector settings, "off" forces point sampling, and
            "auto" or "atom_patch" enables atom-centered local-potential
            patches while leaving nonlocal projectors on the stable coarse
            representation.
        fine_subgrid: Fine-grid samples per coarse-grid axis for unified modes.
        fine_grid_radius_factor: Patch radius multiplier for unified modes.
        local_subgrid: Number of cell-average samples per axis for the local
            pseudopotential. The default 1 keeps point sampling.
        local_mode: "cell_average" to average every coarse cell or "patch" to
            average only near each atom and keep point sampling elsewhere.
        local_patch_radius_factor: Patch radius multiplier applied to each
            local pseudopotential radius in patch mode.
        projector_subgrid: Number of cell-average samples per axis for
            nonlocal projectors. The default 1 keeps point sampling.
        projector_mode: "cell_average" for coarse-cell averaging or "patch"
            for atom-centered fine-grid projector integration.
        projector_patch_radius_factor: Patch radius multiplier applied to each
            projector radius in patch mode.

    Returns:
        Tuple (energy, forces) where energy is in Hartree and forces are in
        Hartree/Bohr. Forces are currently zeros in this implementation.
    """
    if fine_grid_mode not in (None, "off", "auto", "atom_patch"):
        raise ValueError("fine_grid_mode must be None, 'off', 'auto', or 'atom_patch'")
    if fine_subgrid < 1:
        raise ValueError("fine_subgrid must be >= 1")

    if fine_grid_mode == "off":
        local_subgrid = 1
        local_mode = "cell_average"
        projector_subgrid = 1
        projector_mode = "cell_average"
    elif fine_grid_mode in ("auto", "atom_patch"):
        local_subgrid = fine_subgrid
        local_mode = "patch"
        local_patch_radius_factor = fine_grid_radius_factor
        projector_subgrid = 1
        projector_mode = "cell_average"

    zion = jnp.asarray([p["zion"] for p in pseudos])
    rloc = jnp.asarray([p["rloc"] for p in pseudos])
    c = jnp.asarray([p["c"] for p in pseudos])
    
    n_electrons = jnp.sum(jnp.asarray([p["q"] for p in pseudos]))
    n_bands = int(jnp.ceil(n_electrons / 2.0))
    occ = jnp.zeros((n_bands,))
    rem = n_electrons
    for i in range(n_bands):
        val = jnp.minimum(2.0, rem)
        occ = occ.at[i].set(val)
        rem -= val

    V_loc = build_local_potential(
        coords,
        grid.coords,
        zion,
        rloc,
        c,
        spacing=grid.spacing,
        local_subgrid=local_subgrid,
        local_mode=local_mode,
        local_patch_radius_factor=local_patch_radius_factor,
    )
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
        projector_subgrid=projector_subgrid,
        projector_mode=projector_mode,
        projector_patch_radius_factor=projector_patch_radius_factor,
    )
    
    ion_e = ion_ion_energy(coords, zion)
    E_tot = total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid.volume_element, ion_e)
    return E_tot, jnp.zeros_like(coords)
