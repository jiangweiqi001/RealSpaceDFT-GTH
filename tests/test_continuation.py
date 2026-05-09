"""Tests for density continuation / interpolation."""

import unittest

import jax.numpy as jnp


class ContinuationInterpolateTest(unittest.TestCase):
    def test_interpolate_preserves_electron_count_same_box(self):
        from JaxDFT.src.continuation import interpolate_rho_trilinear
        from JaxDFT.src.hamiltonian import create_grid

        box = 4.0
        n_elec = 10.0
        grid_c = create_grid(1.0, [box, box, box])
        grid_f = create_grid(0.5, [box, box, box])
        npts_c = grid_c.shape[0] * grid_c.shape[1] * grid_c.shape[2]
        rho_c = jnp.full(grid_c.shape, n_elec / (npts_c * float(grid_c.volume_element)), dtype=grid_c.coords.dtype)
        rho_f = interpolate_rho_trilinear(grid_c, rho_c, grid_f, n_elec)
        int_f = float(jnp.sum(rho_f) * grid_f.volume_element)
        self.assertAlmostEqual(n_elec, int_f, places=4)


if __name__ == "__main__":
    unittest.main()
