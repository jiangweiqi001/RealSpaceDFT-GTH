import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass


def _periodic_table_Z(sym):
    table = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
    }
    return table[sym]


@dataclass
class GTHLocal:
    zion: float
    rloc: float
    c: np.ndarray


@dataclass
class GTHProjector:
    l: int
    r: float
    h: np.ndarray
    poly: np.ndarray


@dataclass
class GTHPP:
    symbol: str
    q: int
    local: GTHLocal
    projectors: list


class GTHParser:
    @staticmethod
    def parse(text):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        header = lines[0].split()
        symbol = header[0]
        q = int(lines[1].split()[0])
        local_tokens = lines[2].split()
        rloc = float(local_tokens[0])
        n_gauss = int(local_tokens[1])
        c = np.array([float(x) for x in local_tokens[2:2 + n_gauss]], dtype=float)
        zion = _periodic_table_Z(symbol) - q
        local = GTHLocal(zion=zion, rloc=rloc, c=c)
        projectors = []
        idx = 3
        if idx < len(lines):
            try:
                n_channels_tokens = lines[idx].split()
                n_channels = int(n_channels_tokens[0])
                idx += 1
            except Exception:
                n_channels = 0
        else:
            n_channels = 0
        l_cur = 0
        for _ in range(n_channels):
            tokens = lines[idx].split()
            r = float(tokens[0])
            n_proj = int(tokens[1])
            idx += 1
            hvals = []
            for _ in range(n_proj):
                toks = lines[idx].split()
                hvals.append(float(toks[0]))
                idx += 1
            h = np.array(hvals, dtype=float)
            poly = np.arange(n_proj, dtype=int)
            projectors.append(GTHProjector(l=l_cur, r=r, h=h, poly=poly))
            l_cur += 1
        return GTHPP(symbol=symbol, q=q, local=local, projectors=projectors)


class Grid3D:
    def __init__(self, box_origin, box_lengths, shape):
        self.origin = np.array(box_origin, dtype=float)
        self.lengths = np.array(box_lengths, dtype=float)
        self.shape = tuple(int(n) for n in shape)
        self.h = self.lengths / (np.array(self.shape) - 1)
        x = np.linspace(self.origin[0], self.origin[0] + self.lengths[0], self.shape[0])
        y = np.linspace(self.origin[1], self.origin[1] + self.lengths[1], self.shape[1])
        z = np.linspace(self.origin[2], self.origin[2] + self.lengths[2], self.shape[2])
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")
        interior_mask = np.ones(self.shape, dtype=bool)
        interior_mask[[0, -1], :, :] = False
        interior_mask[:, [0, -1], :] = False
        interior_mask[:, :, [0, -1]] = False
        self.interior_mask = interior_mask
        idx = np.arange(np.prod(self.shape)).reshape(self.shape)
        self.interior_idx = idx[self.interior_mask]
        self.n_interior = self.interior_idx.size
        self.coords_interior = np.vstack(
            (self.X.flatten()[self.interior_idx],
             self.Y.flatten()[self.interior_idx],
             self.Z.flatten()[self.interior_idx])
        ).T
        self.L = self._build_laplacian_4th_order()

    def _d2_1d_4th(self, n, h):
        main = (-2.5) * np.ones(n)
        off1 = (4.0 / 3.0) * np.ones(n - 1)
        off2 = (-1.0 / 12.0) * np.ones(n - 2)
        diags = [
            (main, 0),
            (off1, 1),
            (off1, -1),
            (off2, 2),
            (off2, -2),
        ]
        D2 = sp.diags([d for d, _ in diags], [k for _, k in diags], shape=(n, n), format="csr")
        return D2 / (h * h)

    def _build_laplacian_4th_order(self):
        nx, ny, nz = self.shape
        nx_i = nx - 2
        ny_i = ny - 2
        nz_i = nz - 2
        D2x = self._d2_1d_4th(nx_i, self.h[0])
        D2y = self._d2_1d_4th(ny_i, self.h[1])
        D2z = self._d2_1d_4th(nz_i, self.h[2])
        Ix = sp.eye(nx_i, format="csr")
        Iy = sp.eye(ny_i, format="csr")
        Iz = sp.eye(nz_i, format="csr")
        Lx = sp.kron(sp.kron(D2x, Iy), Iz, format="csr")
        Ly = sp.kron(sp.kron(Ix, D2y), Iz, format="csr")
        Lz = sp.kron(sp.kron(Ix, Iy), D2z, format="csr")
        return Lx + Ly + Lz

    def volume_element(self):
        return float(np.prod(self.h))


class Functional:
    @staticmethod
    def lda_exchange_vxc(rho):
        const = (3.0 / np.pi) ** (1.0 / 3.0)
        vx = -const * np.power(rho + 1e-20, 1.0 / 3.0)
        ex = 0.75 * vx * rho
        return ex, vx

    @staticmethod
    def lda_correlation_pz81(rho):
        a = 0.0310907
        b = -0.048
        c = 0.0020
        d = -0.0116
        rs = (3.0 / (4.0 * np.pi * (rho + 1e-30))) ** (1.0 / 3.0)
        def f(rs):
            return a * (np.log(rs) + b * rs + c * rs * np.log(rs) + d)
        ec = f(rs)
        d_ec_drho = a * (1.0 / rs + b + c * (1.0 + np.log(rs))) * (-rs / (3.0 * rho + 1e-30))
        vc = ec + rho * d_ec_drho
        return ec, vc

    @staticmethod
    def lda_xc(rho):
        ex, vx = Functional.lda_exchange_vxc(rho)
        ec, vc = Functional.lda_correlation_pz81(rho)
        eps_xc = ex + ec
        v_xc = vx + vc
        return eps_xc, v_xc


def gth_local_potential_value(r, zion, rloc, c):
    if zion != 0.0:
        t = r / (np.sqrt(2.0) * rloc)
        v_coul = -zion * (2.0 / np.sqrt(np.pi)) * np.exp(-t * t) / r
        v_coul = np.where(r > 1e-12, v_coul, 0.0)
    else:
        v_coul = 0.0
    x = r / rloc
    gauss = np.exp(-0.5 * x * x)
    poly = np.zeros_like(r)
    if c.size > 0:
        poly = poly + c[0]
    if c.size > 1:
        poly = poly + c[1] * (x * x)
    if c.size > 2:
        poly = poly + c[2] * (x ** 4)
    if c.size > 3:
        poly = poly + c[3] * (x ** 6)
    return v_coul + gauss * poly


def gth_local_dV_dr(r, zion, rloc, c):
    if zion != 0.0:
        t = r / (np.sqrt(2.0) * rloc)
        dvdr_coul = -zion * (2.0 / np.sqrt(np.pi)) * np.exp(-t * t) * (-1.0 / (r * r) - (2.0 * t) / (r * np.sqrt(2.0) * rloc))
        dvdr_coul = np.where(r > 1e-12, dvdr_coul, 0.0)
    else:
        dvdr_coul = 0.0
    x = r / rloc
    gauss = np.exp(-0.5 * x * x)
    poly = 0.0
    dpoly_dx = 0.0
    if c.size > 0:
        poly += c[0]
    if c.size > 1:
        poly += c[1] * x * x
        dpoly_dx += 2.0 * c[1] * x
    if c.size > 2:
        poly += c[2] * (x ** 4)
        dpoly_dx += 4.0 * c[2] * (x ** 3)
    if c.size > 3:
        poly += c[3] * (x ** 6)
        dpoly_dx += 6.0 * c[3] * (x ** 5)
    dgauss_dx = -x * gauss
    dVdx = dgauss_dx * poly + gauss * dpoly_dx
    dvdr_gauss = dVdx / rloc
    return dvdr_coul + dvdr_gauss


class DFTEngine:
    def __init__(self, grid: Grid3D, atoms, pseudos, electrons=None):
        self.grid = grid
        self.atoms = np.array(atoms, dtype=float)
        self.pseudos = pseudos
        self.n_atoms = len(atoms)
        self.interior_coords = grid.coords_interior
        self.w = grid.volume_element()
        if electrons is None:
            q_sum = sum([pp.q for pp in pseudos])
            self.electrons = q_sum
        else:
            self.electrons = electrons
        self.n_bands = int(np.ceil(self.electrons / 2.0))
        self.occ = np.zeros(self.n_bands)
        remaining = self.electrons
        for i in range(self.n_bands):
            occ_i = min(2.0, remaining)
            self.occ[i] = occ_i
            remaining -= occ_i
        self.V_loc = self.build_local_potential()
        self.rho = np.zeros(self.grid.n_interior)
        self.V_H = np.zeros_like(self.rho)
        self.v_xc = np.zeros_like(self.rho)
        self.eps_xc = np.zeros_like(self.rho)
        self.projectors = self._build_projectors()

    def build_local_potential(self):
        V = np.zeros(self.grid.n_interior)
        for a, pp in enumerate(self.pseudos):
            rvec = self.interior_coords - self.atoms[a]
            r = np.linalg.norm(rvec, axis=1)
            V += gth_local_potential_value(r, pp.local.zion, pp.local.rloc, pp.local.c)
        return V

    def _build_projectors(self):
        projs = []
        for a, pp in enumerate(self.pseudos):
            for pj in pp.projectors:
                if pj.l != 0:
                    continue
                rvec = self.interior_coords - self.atoms[a]
                r = np.linalg.norm(rvec, axis=1)
                x = r / pj.r
                radial = np.exp(-x * x)
                polys = [radial]
                if pj.h.size > 1:
                    polys.append((x * x) * radial)
                if pj.h.size > 2:
                    polys.append((x ** 4) * radial)
                if pj.h.size > 3:
                    polys.append((x ** 6) * radial)
                for k, hk in enumerate(pj.h):
                    if k < len(polys):
                        projs.append({
                            "atom_index": a,
                            "l": pj.l,
                            "r": pj.r,
                            "h": hk,
                            "vec": polys[k].astype(float),
                        })
        return projs

    def apply_nonlocal(self, psi):
        if not self.projectors:
            return np.zeros_like(psi)
        out = np.zeros_like(psi)
        for P in self.projectors:
            coeff = self.w * np.dot(P["vec"], psi)
            out += P["h"] * coeff * P["vec"]
        return out

    def laplacian(self, psi):
        return self.grid.L @ psi

    def assemble_operator(self, V_eff):
        L = self.grid.L
        diagV = sp.diags(V_eff, 0, format="csr")
        def matvec(x):
            y = -0.5 * (L @ x) + diagV @ x + self.apply_nonlocal(x)
            return y
        return spla.LinearOperator((self.grid.n_interior, self.grid.n_interior), matvec=matvec, dtype=float)

    def solve_poisson(self, rho):
        rhs = -4.0 * np.pi * rho
        Vh = spla.cg(self.grid.L, rhs, x0=self.V_H.copy(), maxiter=200, rtol=1e-8)[0]
        return Vh

    def scf(self, maxiter=30, mix=0.3, k_eigs=None):
        if k_eigs is None:
            k_eigs = self.n_bands
        V_eff = self.V_loc.copy()
        for _ in range(maxiter):
            H = self.assemble_operator(V_eff)
            vals, vecs = spla.eigsh(H, k=k_eigs, which="SA", tol=1e-6)
            for i in range(k_eigs):
                norm = np.sqrt(self.w * np.dot(vecs[:, i], vecs[:, i]))
                vecs[:, i] /= (norm + 1e-30)
            rho_new = np.sum((vecs[:, :k_eigs] ** 2) * self.occ[:k_eigs], axis=1)
            rho = mix * rho_new + (1.0 - mix) * self.rho
            V_H = self.solve_poisson(rho)
            eps_xc, v_xc = Functional.lda_xc(rho)
            V_eff = self.V_loc + V_H + v_xc
            self.rho = rho
            self.V_H = V_H
            self.v_xc = v_xc
            self.eps_xc = eps_xc
            self.band_energies = vals
            self.band_vectors = vecs
        return self.total_energy()

    def kinetic_energy(self):
        T = 0.0
        for i in range(self.n_bands):
            psi = self.band_vectors[:, i]
            T_i = -0.5 * self.w * np.dot(psi, self.grid.L @ psi)
            T += self.occ[i] * T_i
        return T

    def nonlocal_energy(self):
        E = 0.0
        if not self.projectors:
            return 0.0
        for i in range(self.n_bands):
            psi = self.band_vectors[:, i]
            Vnl_psi = self.apply_nonlocal(psi)
            E += self.occ[i] * self.w * np.dot(psi, Vnl_psi)
        return E

    def local_energy(self):
        return self.w * np.dot(self.rho, self.V_loc)

    def hartree_energy(self):
        return 0.5 * self.w * np.dot(self.rho, self.V_H)

    def xc_energy(self):
        return self.w * np.dot(self.eps_xc, np.ones_like(self.eps_xc))

    def ion_ion_energy(self):
        E = 0.0
        for a in range(self.n_atoms):
            za = self.pseudos[a].local.zion
            if za == 0.0:
                continue
            for b in range(a + 1, self.n_atoms):
                zb = self.pseudos[b].local.zion
                if zb == 0.0:
                    continue
                r = np.linalg.norm(self.atoms[a] - self.atoms[b])
                if r > 1e-12:
                    E += za * zb / r
        return E

    def total_energy(self):
        return self.kinetic_energy() + self.local_energy() + self.nonlocal_energy() + self.hartree_energy() + self.xc_energy() + self.ion_ion_energy()

    def calculate_forces(self):
        forces = np.zeros_like(self.atoms)
        rho = self.rho
        for a, pp in enumerate(self.pseudos):
            rvec = self.interior_coords - self.atoms[a]
            r = np.linalg.norm(rvec, axis=1)
            dvdr = gth_local_dV_dr(r, pp.local.zion, pp.local.rloc, pp.local.c)
            unit = np.where(r[:, None] > 1e-12, rvec / r[:, None], 0.0)
            f_loc = self.w * np.sum((rho * dvdr)[:, None] * unit, axis=0)
            forces[a] += f_loc
        for P in self.projectors:
            a = P["atom_index"]
            rvec = self.interior_coords - self.atoms[a]
            r = np.linalg.norm(rvec, axis=1)
            x = r / P["r"]
            radial = np.exp(-x * x)
            beta = radial
            coeff_sum = 0.0
            for i in range(self.n_bands):
                psi = self.band_vectors[:, i]
                coeff_sum += self.occ[i] * self.w * np.dot(beta, psi)
            dbeta_dr = radial * (-2.0 * x) / P["r"]
            unit = np.where(r[:, None] > 1e-12, rvec / r[:, None], 0.0)
            grad_beta = (dbeta_dr[:, None]) * unit
            f_nl = 2.0 * P["h"] * coeff_sum * self.w * np.sum(grad_beta * beta[:, None], axis=0)
            forces[a] += f_nl
        for a in range(self.n_atoms):
            za = self.pseudos[a].local.zion
            if za == 0.0:
                continue
            for b in range(self.n_atoms):
                if a == b:
                    continue
                zb = self.pseudos[b].local.zion
                if zb == 0.0:
                    continue
                Rab = self.atoms[a] - self.atoms[b]
                r = np.linalg.norm(Rab)
                if r > 1e-12:
                    forces[a] += za * zb * Rab / (r ** 3)
        return -forces


H_POTENTIAL_STR = """
H GTH-PADE-q1 GTH-LDA-q1 GTH-PADE GTH-LDA
    1
     0.20000000    2    -4.18023680     0.72507482
    0
"""

He_POTENTIAL_STR = """
He GTH-PADE-q2 GTH-LDA-q2 GTH-PADE GTH-LDA
    2
     0.20000000    2    -9.11202340     1.69836797
    0
"""


def random_cluster(center, Rc, N_neighbors):
    rng = np.random.default_rng(42)
    pts = [np.array(center, dtype=float)]
    for _ in range(N_neighbors):
        u = rng.random()
        r = Rc * (u ** (1.0 / 3.0))
        theta = np.arccos(1 - 2 * rng.random())
        phi = 2 * np.pi * rng.random()
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        pts.append(np.array(center, dtype=float) + np.array([x, y, z], dtype=float))
    return np.array(pts)


def main():
    center = [0.0, 0.0, 0.0]
    Rc = 2.0
    N_neighbors = 2
    atoms = random_cluster(center, Rc, N_neighbors)
    pseudos = [GTHParser.parse(He_POTENTIAL_STR) for _ in range(len(atoms))]
    L = np.array([16.0, 16.0, 16.0])
    shape = (28, 28, 28)
    origin = -L / 2.0
    grid = Grid3D(origin, L, shape)
    engine = DFTEngine(grid, atoms, pseudos)
    E = engine.scf(maxiter=20, mix=0.4, k_eigs=engine.n_bands)
    F = engine.calculate_forces()
    print("Total energy (Ha):", E)
    print("Atomic positions (Bohr):")
    print(atoms)
    print("Forces (Ha/Bohr):")
    print(F)


if __name__ == "__main__":
    main()