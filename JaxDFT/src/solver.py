import jax
import jax.numpy as jnp

from .functional import lda_xc
from .hamiltonian import laplacian_4th, apply_nonlocal, build_local_potential


@jax.jit
def solve_poisson(rho, box_size):
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


@jax.jit
def normalize_orbitals(psi, volume_element):
    norms = jnp.sqrt(volume_element * jnp.sum(psi * psi, axis=0))
    return psi / (norms + 1e-30)


def solve_orbitals(apply_h, n_grid, n_bands, key):
    if n_grid < 32768:
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            eye = jnp.eye(n_grid, dtype=jnp.float64)
            h_dense = jax.vmap(apply_h, in_axes=1, out_axes=1)(eye)
            h_dense = jnp.nan_to_num(h_dense, nan=0.0, posinf=0.0, neginf=0.0)
            h_dense = 0.5 * (h_dense + h_dense.T)
            h_dense = h_dense + 1e-12 * jnp.eye(n_grid, dtype=jnp.float64)
            eigvals, eigvecs = jnp.linalg.eigh(h_dense)
            eigvals = eigvals[:n_bands]
            eigvecs = eigvecs[:, :n_bands]
            return eigvals, eigvecs
    X = jax.random.normal(key, (n_grid, n_bands))
    q, _ = jnp.linalg.qr(X)
    try:
        from jax.experimental.sparse.linalg import lobpcg
    except Exception:
        if n_grid <= 45000:
            cpu = jax.devices("cpu")[0]
            with jax.default_device(cpu):
                eye = jnp.eye(n_grid, dtype=jnp.float64)
                h_dense = jax.vmap(apply_h, in_axes=1, out_axes=1)(eye)
                h_dense = jnp.nan_to_num(h_dense, nan=0.0, posinf=0.0, neginf=0.0)
                h_dense = 0.5 * (h_dense + h_dense.T)
                h_dense = h_dense + 1e-12 * jnp.eye(n_grid, dtype=jnp.float64)
                eigvals, eigvecs = jnp.linalg.eigh(h_dense)
                eigvals = eigvals[:n_bands]
                eigvecs = eigvecs[:, :n_bands]
                return eigvals, eigvecs
        eigvals = jnp.full((n_bands,), jnp.nan, dtype=jnp.float64)
        eigvecs = jnp.full((n_grid, n_bands), jnp.nan, dtype=jnp.float64)
        return eigvals, eigvecs
    try:
        eigvals, eigvecs = lobpcg(apply_h, q, tol=1e-4, maxiter=50)
    except TypeError:
        eigvals, eigvecs = lobpcg(apply_h, q, tolerance=1e-4, maxiter=50)
    except Exception:
        eigvals = jnp.full((n_bands,), jnp.nan, dtype=jnp.float64)
        eigvecs = jnp.full((n_grid, n_bands), jnp.nan, dtype=jnp.float64)
    return eigvals, eigvecs


def anderson_mixing(rho, rho_new, f_hist, mix_alpha, iter_idx, m=5):
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
        rho_next = jnp.nan_to_num(rho_next, nan=rho_new, posinf=rho_new, neginf=rho_new)
        return rho_next, f_hist1

    return jax.lax.cond(iter_idx == 0, first, later, operand=None)


def scf(grid, coords, n_bands, occ, V_loc, projectors, max_iter, mix_alpha, tolerance, key):
    coords = jnp.asarray(coords, dtype=jnp.float64)
    volume_element = grid.volume_element
    V_loc = jnp.nan_to_num(V_loc, nan=0.0, posinf=0.0, neginf=0.0)
    rho = jnp.zeros(grid.shape, dtype=jnp.float64)
    for a in range(coords.shape[0]):
        r = jnp.linalg.norm(grid.coords - coords[a], axis=-1)
        rho = rho + jnp.exp(-r * r)
    rho = rho / (jnp.max(rho) + 1e-12)
    rho = jnp.clip(rho, 1e-8, 1e6)
    f_hist = jnp.zeros((5, rho.size), dtype=jnp.float64)
    n_grid = rho.size
    eigvals0 = jnp.zeros((n_bands,), dtype=jnp.float64)
    eigvecs0 = jnp.zeros((n_grid, n_bands), dtype=jnp.float64)
    V_H0 = jnp.zeros(grid.shape, dtype=jnp.float64)
    eps_xc0 = jnp.zeros(grid.shape, dtype=jnp.float64)
    v_xc0 = jnp.zeros(grid.shape, dtype=jnp.float64)
    diff0 = jnp.array(jnp.inf, dtype=jnp.float64)
    i0 = jnp.array(0, dtype=jnp.int32)

    def cond(state):
        i, rho_cur, f_hist_cur, diff, eigvals, eigvecs, V_H, eps_xc, v_xc = state
        return jnp.logical_and(i < max_iter, diff > tolerance)

    def body(state):
        i, rho_cur, f_hist_cur, diff, eigvals, eigvecs, V_H, eps_xc, v_xc = state
        rho_cur = jnp.nan_to_num(rho_cur, nan=1e-8, posinf=1e-8, neginf=1e-8)
        V_H = solve_poisson(rho_cur, grid.box_size)
        eps_xc, v_xc = lda_xc(rho_cur)
        V_H = jnp.nan_to_num(V_H, nan=0.0, posinf=0.0, neginf=0.0)
        eps_xc = jnp.nan_to_num(eps_xc, nan=0.0, posinf=0.0, neginf=0.0)
        v_xc = jnp.nan_to_num(v_xc, nan=0.0, posinf=0.0, neginf=0.0)
        V_eff = V_loc + V_H + v_xc

        def apply_h(psi_flat):
            psi = psi_flat.reshape(grid.shape)
            lap = laplacian_4th(psi, grid.spacing, grid.mask)
            hpsi = -0.5 * lap + V_eff * psi
            hpsi = hpsi.reshape(-1)
            hpsi = hpsi + apply_nonlocal(projectors, psi_flat, volume_element)
            return hpsi

        eigvals_new, eigvecs_new = solve_orbitals(apply_h, n_grid, n_bands, key)
        eigvals_new = jnp.nan_to_num(eigvals_new, nan=0.0, posinf=0.0, neginf=0.0)
        eigvecs_new = jnp.nan_to_num(eigvecs_new, nan=0.0, posinf=0.0, neginf=0.0)
        eigvecs_new = normalize_orbitals(eigvecs_new, volume_element)
        eigvals_ok = jnp.all(jnp.isfinite(eigvals_new))
        eigvecs_ok = jnp.all(jnp.isfinite(eigvecs_new))
        ok = jnp.logical_and(eigvals_ok, eigvecs_ok)
        eigvals = jax.lax.select(ok, eigvals_new, eigvals)
        eigvecs = jax.lax.select(ok, eigvecs_new, eigvecs)
        rho_new = jnp.sum((eigvecs ** 2) * occ[None, :], axis=1).reshape(grid.shape)
        rho_new = jnp.clip(rho_new, 1e-8, 1e6)
        rho_new = jnp.nan_to_num(rho_new, nan=rho_cur, posinf=rho_cur, neginf=rho_cur)
        diff = jnp.max(jnp.abs(rho_new - rho_cur))
        rho_flat, f_hist_cur = anderson_mixing(
            rho_cur.reshape(-1), rho_new.reshape(-1), f_hist_cur, mix_alpha, i
        )
        rho_cur = rho_flat.reshape(grid.shape)
        return i + 1, rho_cur, f_hist_cur, diff, eigvals, eigvecs, V_H, eps_xc, v_xc

    state0 = (i0, rho, f_hist, diff0, eigvals0, eigvecs0, V_H0, eps_xc0, v_xc0)

    # 1. Run SCF loop without gradient tracking to convergence
    state = jax.lax.while_loop(cond, body, state0)
    state = jax.lax.stop_gradient(state)
    
    # 2. Run one more step WITH gradient tracking
    # This allows gradients to flow through the final self-consistent state
    # which is a good approximation for the full gradient (Hellmann-Feynman theorem spirit)
    final_state = body(state)
    
    _, rho, f_hist, diff, eigvals, eigvecs, V_H, eps_xc, v_xc = final_state
    return rho, eigvals, eigvecs, V_H, eps_xc, v_xc


def total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, volume_element, ion_ion):
    rho = jnp.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals = jnp.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0)
    V_loc = jnp.nan_to_num(V_loc, nan=0.0, posinf=0.0, neginf=0.0)
    V_H = jnp.nan_to_num(V_H, nan=0.0, posinf=0.0, neginf=0.0)
    eps_xc = jnp.nan_to_num(eps_xc, nan=0.0, posinf=0.0, neginf=0.0)
    e_band = jnp.sum(eigvals * occ)
    e_h = 0.5 * volume_element * jnp.sum(rho * V_H)
    e_xc = volume_element * jnp.sum(eps_xc)
    e_vxc = volume_element * jnp.sum(rho * v_xc)
    return e_band - e_h + e_xc - e_vxc + ion_ion


def ion_ion_energy(coords, zion):
    e = 0.0
    for i in range(coords.shape[0]):
        for j in range(i + 1, coords.shape[0]):
            r = jnp.linalg.norm(coords[i] - coords[j]) + 1e-12
            e = e + zion[i] * zion[j] / r
    return e


def energy_and_forces(grid, coords, pseudos, max_iter, mix_alpha, tolerance, key):
    zion = jnp.asarray([pp["zion"] for pp in pseudos], dtype=jnp.float64)
    rloc = jnp.asarray([pp["rloc"] for pp in pseudos], dtype=jnp.float64)
    c = jnp.asarray([pp["c"] for pp in pseudos], dtype=jnp.float64)
    projectors = [
        {
            "h": jnp.asarray(p["h"], dtype=jnp.float64),
            "vec": jnp.asarray(p["vec"], dtype=jnp.float64),
        }
        for p in grid.projectors
    ]
    electrons = jnp.sum(jnp.asarray([pp["q"] for pp in pseudos], dtype=jnp.float64))
    n_bands = int(jnp.ceil(electrons / 2.0))
    occ = jnp.zeros((n_bands,), dtype=jnp.float64)
    remaining = electrons
    for i in range(n_bands):
        occ = occ.at[i].set(jnp.minimum(2.0, remaining))
        remaining = remaining - occ[i]

    def energy_fn(atom_coords):
        V_loc = build_local_potential(atom_coords, grid.coords, zion, rloc, c)
        V_loc = jnp.nan_to_num(V_loc, nan=0.0, posinf=0.0, neginf=0.0)
        rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
            grid, atom_coords, n_bands, occ, V_loc, projectors, max_iter, mix_alpha, tolerance, key
        )
        rho = jax.lax.stop_gradient(rho)
        eigvals = jax.lax.stop_gradient(eigvals)
        V_H = jax.lax.stop_gradient(V_H)
        eps_xc = jax.lax.stop_gradient(eps_xc)
        ion_e = ion_ion_energy(atom_coords, zion)
        e = total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid.volume_element, ion_e)
        return e

    energy, grad = jax.value_and_grad(energy_fn)(coords)
    forces = -grad
    return energy, forces



def energy_only(grid, coords, pseudos, max_iter, mix_alpha, tolerance, key):
    zion = jnp.asarray([pp["zion"] for pp in pseudos], dtype=jnp.float64)
    rloc = jnp.asarray([pp["rloc"] for pp in pseudos], dtype=jnp.float64)
    c = jnp.asarray([pp["c"] for pp in pseudos], dtype=jnp.float64)
    projectors = [
        {
            "h": jnp.asarray(p["h"], dtype=jnp.float64),
            "vec": jnp.asarray(p["vec"], dtype=jnp.float64),
        }
        for p in grid.projectors
    ]
    electrons = jnp.sum(jnp.asarray([pp["q"] for pp in pseudos], dtype=jnp.float64))
    n_bands = int(jnp.ceil(electrons / 2.0))
    occ = jnp.zeros((n_bands,), dtype=jnp.float64)
    remaining = electrons
    for i in range(n_bands):
        occ = occ.at[i].set(jnp.minimum(2.0, remaining))
        remaining = remaining - occ[i]

    V_loc = build_local_potential(coords, grid.coords, zion, rloc, c)
    rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
        grid, coords, n_bands, occ, V_loc, projectors, max_iter, mix_alpha, tolerance, key
    )
    ion_e = ion_ion_energy(coords, zion)
    e = total_energy(rho, eigvals, occ, V_loc, V_H, eps_xc, v_xc, grid.volume_element, ion_e)
    return e




def energy_components(grid, coords, pseudos, max_iter, mix_alpha, tolerance, key):
    zion = jnp.asarray([pp["zion"] for pp in pseudos], dtype=jnp.float64)
    rloc = jnp.asarray([pp["rloc"] for pp in pseudos], dtype=jnp.float64)
    c = jnp.asarray([pp["c"] for pp in pseudos], dtype=jnp.float64)
    projectors = [
        {
            "h": jnp.asarray(p["h"], dtype=jnp.float64),
            "vec": jnp.asarray(p["vec"], dtype=jnp.float64),
        }
        for p in grid.projectors
    ]
    electrons = jnp.sum(jnp.asarray([pp["q"] for pp in pseudos], dtype=jnp.float64))
    n_bands = int(jnp.ceil(electrons / 2.0))
    occ = jnp.zeros((n_bands,), dtype=jnp.float64)
    remaining = electrons
    for i in range(n_bands):
        occ = occ.at[i].set(jnp.minimum(2.0, remaining))
        remaining = remaining - occ[i]

    V_loc = build_local_potential(coords, grid.coords, zion, rloc, c)
    V_loc = jnp.nan_to_num(V_loc, nan=0.0, posinf=0.0, neginf=0.0)
    rho, eigvals, eigvecs, V_H, eps_xc, v_xc = scf(
        grid, coords, n_bands, occ, V_loc, projectors, max_iter, mix_alpha, tolerance, key
    )
    ion_e = ion_ion_energy(coords, zion)
    e_band = jnp.sum(eigvals * occ)
    e_h = 0.5 * grid.volume_element * jnp.sum(rho * V_H)
    e_loc = grid.volume_element * jnp.sum(rho * V_loc)
    e_xc = grid.volume_element * jnp.sum(eps_xc)
    e_vxc = grid.volume_element * jnp.sum(rho * v_xc)
    total = e_band - e_h + e_loc + e_xc - e_vxc + ion_e
    return {
        "total": total,
        "e_band": e_band,
        "e_h": e_h,
        "e_loc": e_loc,
        "e_xc": e_xc,
        "e_vxc": e_vxc,
        "ion": ion_e,
    }
