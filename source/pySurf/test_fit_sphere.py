import numpy as np

from pySurf.fit_sphere import sphere_error3, fit_sphere


def _random_sphere_points(center, radius, n=2000, noise_sigma=0.0, seed=0):
    rng = np.random.default_rng(seed)
    # Uniform on sphere: sample normal vectors and normalize
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    pts = center + radius * v
    if noise_sigma:
        pts = pts + rng.normal(scale=noise_sigma, size=pts.shape)
    return pts


def test_sphere_error3_perfect_sphere():
    center = np.array([1.0, -2.0, 0.5])
    radius = 10.0
    pts = _random_sphere_points(center, radius, n=500, noise_sigma=0.0, seed=1)

    fom, residuals, rad = sphere_error3(center, pts, retall=True)

    assert np.isclose(fom, 0.0, atol=1e-12)
    assert np.isclose(rad, radius, atol=1e-12)
    assert residuals.shape == (pts.shape[0], 3)
    assert np.allclose(residuals[:, 2], 0.0, atol=1e-12)


def test_fit_sphere_recovers_center_and_radius():
    true_center = np.array([0.8, -1.2, 2.5])
    true_radius = 7.5
    noise = 0.01
    pts = _random_sphere_points(true_center, true_radius, n=1500, noise_sigma=noise, seed=2)

    # Start from an offset guess
    guess = true_center + np.array([0.6, -0.4, 0.3])

    residuals, center_fit, radius_fit = fit_sphere(
        pts,
        guess,
        zscale=1.0,
        keepnan=False,
        return_radius=True,
        verbose=False,
        method="Nelder-Mead",
        options={"maxiter": 3000},
    )

    # With moderate noise, the fitted center should be close
    assert np.linalg.norm(center_fit - true_center) < 0.05
    assert abs(radius_fit - true_radius) < 0.05

    # RMS residual should be on the order of the injected noise
    rms = float(np.sqrt(np.mean(residuals[:, 2] ** 2)))
    assert rms < 0.05


def test_fit_sphere_fixed_radius():
    true_center = np.array([2.0, 1.0, -3.0])
    true_radius = 12.0
    pts = _random_sphere_points(true_center, true_radius, n=1000, noise_sigma=0.02, seed=3)

    guess = true_center + np.array([0.2, 0.2, -0.2])
    residuals, center_fit, radius_fit = fit_sphere(
        pts,
        guess,
        return_radius=True,
        verbose=False,
        method="Nelder-Mead",
        options={"maxiter": 2500, "radius": true_radius},
    )

    assert np.isclose(radius_fit, true_radius, atol=1e-12)
    assert np.linalg.norm(center_fit - true_center) < 0.1
    assert residuals.shape == (pts.shape[0], 3)


def test_fit_sphere_keepnan_preserves_shape_and_nans():
    true_center = np.array([0.0, 0.0, 0.0])
    true_radius = 3.0
    pts = _random_sphere_points(true_center, true_radius, n=500, noise_sigma=0.0, seed=4)

    # Insert NaNs in Z
    pts_with_nan = pts.copy()
    pts_with_nan[::7, 2] = np.nan

    guess = np.array([0.1, -0.1, 0.2])

    residuals, center_fit = fit_sphere(
        pts_with_nan,
        guess,
        keepnan=True,
        verbose=False,
        method="Nelder-Mead",
        options={"maxiter": 1500},
    )

    assert residuals.shape == pts_with_nan.shape
    assert np.isnan(residuals[::7, 2]).all()
    assert np.isfinite(residuals[np.isfinite(residuals[:, 2]), 2]).all()
    assert np.linalg.norm(center_fit - true_center) < 0.05
