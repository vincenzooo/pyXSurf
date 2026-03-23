import numpy as np
import pytest

from pySurf.fit_sphere import sphere_error, fit_sphere


def _random_points_on_sphere(center, radius, n=500, seed=0, noise_sigma=0.0):
    rng = np.random.default_rng(seed)
    # Uniform on sphere: sample normal vectors and normalize
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    pts = center + radius * v
    if noise_sigma:
        pts = pts + rng.normal(scale=noise_sigma, size=pts.shape)
    return pts


def test_sphere_error_exact_zero_rms():
    c = np.array([1.0, -2.0, 3.0])
    r = 10.0
    pts = _random_points_on_sphere(c, r, n=300, seed=1)
    rms, residuals, rfit = sphere_error(c, pts, retall=True)
    assert rms < 1e-12
    assert abs(rfit - r) < 1e-12
    assert residuals.shape == (300, 3)
    # x,y columns are original x,y
    assert np.allclose(residuals[:, 0], pts[:, 0])
    assert np.allclose(residuals[:, 1], pts[:, 1])


def test_sphere_error_noise_rms_matches_scale():
    c = np.array([0.5, 0.25, -1.25])
    r = 50.0
    pts = _random_points_on_sphere(c, r, n=2000, seed=2)

    rng = np.random.default_rng(3)
    # radial noise
    eps = rng.normal(scale=0.02, size=(pts.shape[0],))
    v = pts - c
    v /= np.linalg.norm(v, axis=1)[:, None]
    pts_noisy = pts + eps[:, None] * v

    rms, _res, _rfit = sphere_error(c, pts_noisy, retall=True)
    # RMS should be close to noise std (within a loose tolerance)
    assert 0.012 < rms < 0.03


def test_fit_sphere_no_optimization_keeps_center():
    c_true = np.array([2.0, 3.0, 4.0])
    r = 12.0
    pts = _random_points_on_sphere(c_true, r, n=500, seed=4)

    c0 = np.array([10.0, 10.0, 10.0])
    residuals, c_out = fit_sphere(pts, c0, maxiter=None, verbose=False)
    # default is evaluation only
    assert np.allclose(c_out, c0)
    assert residuals.shape == (pts.shape[0], 3)


def test_fit_sphere_optimization_recovers_center():
    c_true = np.array([5.0, -1.0, 2.0])
    r = 20.0
    pts = _random_points_on_sphere(c_true, r, n=800, seed=5)

    rng = np.random.default_rng(6)
    # add small isotropic noise
    pts = pts + rng.normal(scale=0.05, size=pts.shape)

    c0 = c_true + np.array([3.0, -4.0, 2.5])

    # evaluate FOM at starting guess
    rms0 = sphere_error(c0, pts)

    residuals, c_fit = fit_sphere(
        pts,
        c0,
        maxiter=800,
        method="Nelder-Mead",
        verbose=False,
        options={"xatol": 1e-8, "fatol": 1e-8},
    )

    rms1 = float(np.sqrt(np.mean(residuals[:, 2] ** 2)))
    assert rms1 < rms0
    assert np.linalg.norm(c_fit - c_true) < 0.3


def test_fit_sphere_keepnan_preserves_shape_and_nans():
    c_true = np.array([0.0, 0.0, 0.0])
    r = 5.0
    pts = _random_points_on_sphere(c_true, r, n=200, seed=7)

    pts_with_nan = pts.copy()
    pts_with_nan[::10, 2] = np.nan

    out, c_out = fit_sphere(pts_with_nan, c_true, keepnan=True, verbose=False)
    assert out.shape == pts_with_nan.shape
    assert np.isnan(out[::10, 2]).all()
    # x,y should be preserved for valid points
    valid = ~np.isnan(pts_with_nan[:, 2])
    assert np.allclose(out[valid, 0], pts_with_nan[valid, 0])
    assert np.allclose(out[valid, 1], pts_with_nan[valid, 1])
    assert np.allclose(c_out, c_true)


def test_fit_sphere_zscale_interprets_z_units():
    # x,y in mm; z provided in um; zscale converts um->mm
    c_true_mm = np.array([1.0, 2.0, 10.0])
    r_mm = 3.0

    pts_mm = _random_points_on_sphere(c_true_mm, r_mm, n=600, seed=8)

    pts_mixed = pts_mm.copy()
    pts_mixed[:, 2] *= 1000.0  # um

    c0 = c_true_mm + np.array([0.8, -0.6, 0.4])

    _, c_fit = fit_sphere(pts_mixed, c0, zscale=1000.0, maxiter=600, verbose=False)

    assert np.linalg.norm(c_fit - c_true_mm) < 0.2
