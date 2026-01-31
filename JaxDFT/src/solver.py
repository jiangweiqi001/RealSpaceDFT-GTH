"""Self-consistent field solver for real-space Kohn-Sham DFT in JAX.

All quantities are in atomic units: length in Bohr, energy in Hartree, and
forces in Hartree/Bohr. The SCF loop builds the effective potential, solves
the Kohn-Sham eigenproblem, and mixes the density until convergence.
"""

import jax
import jax.numpy as jnp
from .functional import lda_xc
from .hamiltonian import laplacian_4th, apply_nonlocal, build_local_potential


@jax.jit
def solve_poisson(rho, box_size):
    """Solve the Poisson equation with an FFT-based spectral method.

    The FFT implies periodic boundary conditions on the simulation cell.

    Args:
        rho: Electron density on the grid, shape (nx, ny, nz), in Bohr^-3.
        box_size: Simulation box lengths [Lx, Ly, Lz] in Bohr.

    Returns:
        Hartree potential on the grid, in Hartree.
    """
    nx, ny, nz = rho.shape
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=box_size[0] / (nx - 1))
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=box_size[1] / (ny - 1))
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=box_size[2] / (nz - 1))
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX * KX + KY * KY + KZ * KZ
    rho_k = jnp.fft.fftn(rho)
    v_k = 4.0 * jnp.pi * rho_k / jnp.where(k2 > 0, k2, 1.0)
    v_k = v_k.at[0, 0, 0].set(0.0)
    v = jnp.fft.ifftn(v_k).real
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


def scf(grid, coords, n_bands, occ, V_loc, projectors, max_iter, mix_alpha, tolerance, key):
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
        key: JAX PRNG key (kept for API consistency).

    Returns:
        Tuple (rho, eigvals, eigvecs, V_H, eps_xc, v_xc) where energies are in
        Hartree and densities in Bohr^-3.
    """
    coords = jnp.asarray(coords, dtype=jnp.float32)
    volume_element = grid.volume_element
    
    # 初始密度
    rho = jnp.zeros(grid.shape, dtype=jnp.float32)
    for a in range(coords.shape[0]):
        r = jnp.linalg.norm(grid.coords - coords[a], axis=-1)
        rho = rho + jnp.exp(-2.0 * r**2)
    rho = rho / (jnp.sum(rho) * volume_element) * jnp.sum(occ)

    f_hist = jnp.zeros((5, rho.size), dtype=jnp.float32)
    n_grid = rho.size
    
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
        i, rho_cur, f_hist_cur, diff, _, _, _, _, _ = state
        rho_cur = jnp.clip(rho_cur, 1e-12, None)
        V_H = solve_poisson(rho_cur, grid.box_size)
        eps_xc, v_xc = lda_xc(rho_cur)
        V_eff = V_loc + V_H + v_xc

        def apply_h(psi_flat):
            psi = psi_flat.reshape(grid.shape)
            lap = laplacian_4th(psi, grid.spacing, grid.mask)
            hpsi = -0.5 * lap + V_eff * psi
            return hpsi.reshape(-1)

        # 换回 Dense Solver
        eigvals, eigvecs = solve_orbitals_dense(apply_h, n_grid, n_bands)
        
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


def energy_and_forces(grid, coords, pseudos, max_iter, mix_alpha, tolerance, key):
    """Run SCF and return total energy and forces.

    Args:
        grid: Grid object produced by create_grid.
        coords: Ion coordinates, shape (n_atoms, 3), in Bohr.
        pseudos: List of pseudopotential dictionaries.
        max_iter: Maximum SCF iterations.
        mix_alpha: Anderson mixing strength.
        tolerance: Convergence threshold for density change.
        key: JAX PRNG key.

    Returns:
        Tuple (energy, forces) where energy is in Hartree and forces are in
        Hartree/Bohr. Forces are currently zeros in this implementation.
    """
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

    V_loc = build_local_potential(coords, grid.coords, zion, rloc, c)
    rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
        grid, coords, n_bands, occ, V_loc, [], max_iter, mix_alpha, tolerance, key
    )
    
    ion_e = ion_ion_energy(coords, zion)
    E_tot = total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid.volume_element, ion_e)
    return E_tot, jnp.zeros_like(coords)
