"""
fit_sphere
==========

Sphere fitting utilities for 3D point clouds.

Key functions
-------------
- ``sphere_error``: compute RMS radial error of points w.r.t. a sphere, optionally
  fitting radius analytically for each center.
- ``fit_sphere``: wrapper that handles NaNs, optional Z scaling, and optional
  optimization.

Design notes
------------
In the original ``fit_cylinder.fit_cylinder`` wrapper, optimization is enabled by
placing ``maxiter`` inside the SciPy ``options`` dictionary and then *popping*
that key to decide whether to run the optimizer. That has two drawbacks:

1) It is not intuitive unless you already know SciPy's ``minimize`` API.
2) Popping ``maxiter`` removes it from the dictionary passed to SciPy, so the
   optimizer may not actually be constrained by that limit.

Here, optimization is controlled by an explicit ``maxiter`` argument.
For backwards compatibility, callers may still pass ``options={'maxiter': ...}``,
but the value is read without removing it, and the explicit ``maxiter`` argument
(always) takes precedence.

Conventions
-----------
The *RMS* figure-of-merit is always computed from the orthogonal (radial)
residual magnitude ``|distance(center, point) - radius|``.

For the *sign* of the returned per-point residuals, this module defaults to a
"Z-up" convention that is often more convenient for surface metrology:

- A residual is **positive** if the point is **above** the fitted sphere along
  the global +Z direction, i.e. ``z_point > z_sphere(x_point, y_point)``.
- This remains true whether the fitted cap is "convex" or "concave".

The legacy "inside-positive" convention (matching ``fit_cylinder.cylinder_error3``)
can be requested via ``sign_mode='inside'``.

Units
-----
Like ``fit_cylinder.fit_cylinder``, ``fit_sphere`` supports ``zscale`` to
normalize units before fitting:

- Input points are assumed to be ``[x, y, z]``.
- Internally, points are divided by ``[1, 1, zscale]`` before fitting.
- Output residuals are returned in the original units (z multiplied back).

The implementation does not depend on external project packages (e.g., ``pySurf``),
so it can be used as a standalone utility.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from dataIO.span import span


#suggested by chatGPT, but I prefer to keep the type hinting simple for now and not add a dependency on numpy.typing

# def _as_points(points: ArrayLike) -> np.ndarray:
#     """Convert input to an (N, 3) float array and validate shape."""
#     pts = np.asarray(points, dtype=float)
#     if pts.ndim != 2 or pts.shape[1] != 3:
#         raise ValueError(f"`points` must be an array of shape (N, 3); got {pts.shape}.")
#     return pts

def sphere_error(
    center: ArrayLike = (0.0, 0.0, 0.0),
    pts: ArrayLike = None,
    *,
    radius: Optional[float] = None,
    retall: bool = False,
):
    """
    Calculate RMS surface error for points w.r.t. a sphere.

    Parameters
    ----------
    center
        3-element center vector (cx, cy, cz).
    pts
        pts as an (N, 3) array. NaNs are allowed but will propagate; for
        efficiency, filter NaNs before repeated calls (e.g., inside an optimizer).
    radius
        If provided, use this radius (no radius fit). If ``None``, compute the
        best-fit radius for the given *center* as the mean radial distance.
    retall
        If True, also return residuals and radius.

    Returns
    -------
    fom : float
        RMS of radial residuals.
    residuals : (N, 3) array, optional
        Array ``[x, y, deltaR]`` where ``deltaR = radius - ||p-center||``. Returned if ``retall=True``.
    radius : float, optional
        Radius used (fitted or provided). Returned if ``retall=True``.

    Notes
    -----
    The RMS value is computed from the orthogonal (radial) residual magnitude.

    The returned per-point residuals (3rd column of ``residuals``) 
    residual magnitude is the orthogonal residual magnitude.
    
    For the *sign* of the returned per-point residuals, this module defaults to a
    "Z-up" convention that is often more convenient for surface metrology:

    - A residual is **positive** if the point is **above** the fitted sphere along
    the global +Z direction, i.e. ``z_point > z_sphere(x_point, y_point)``.
    - This remains true whether the fitted cap is "convex" or "concave".
    
    Units
    -----
    Like ``fit_cylinder.fit_cylinder``, ``fit_sphere`` supports ``zscale`` to
    normalize units before fitting:

    - Input points are assumed to be ``[x, y, z]``.
    - Internally, points are divided by ``[1, 1, zscale]`` before fitting.
    - Output residuals are returned in the original units (z multiplied back).

    """
    
    if pts is None:
        raise ValueError("`pts` cannot be None.")
    
    c = np.asarray(center, dtype=float).reshape(3)
    
    # Filter rows with any NaN (most often z). This keeps behavior predictable.
    mask = ~np.isnan(pts).any(axis=1)
    pts = pts[mask]

    if pts.size == 0:
        raise ValueError("No valid (non-NaN) points provided.")

    # Distances from center for each point
    R = np.sqrt(np.nansum((pts - c) ** 2, axis=1))

    if radius is None:
        rad = float(np.nanmean(R)) #best fit radius is mean distance from center
    else:
        rad = float(radius)

    deltaR = rad - R  # bump-positive convention
    fom = float(np.sqrt(np.nansum(deltaR**2) / len(deltaR)))
    
    if not retall:
        return fom

    # residuals shaped like the project convention: [x, y, deltaR]
    x, y, _z = np.hsplit(pts, 3)
    residuals = np.hstack([x, y, deltaR[:, None]])

    if retall:
        return fom, residuals, rad
    else:
        return fom


def fit_sphere(
    pts: ArrayLike,
    center0: ArrayLike,
    *,
    maxiter = None,
    zscale: float = 1.0,
    keepnan: bool = False,
    return_result: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """
    Fit a sphere to a point cloud and return residuals.

    This follows the *fit_cylinder.py* wrapper style:

    - If maxiter is provided (either explicitly or as ``options={'maxiter': N}`` N != 0), an optimization is run.
      Otherwise, the FOM is evaluated at ``center0`` without optimization.
    - ``zscale`` is used to scale Z during the fit so that X/Y and Z share units
      (e.g. X/Y in mm, Z in µm -> use zscale=1000).
    - If ``keepnan=True``, NaNs in the original Z column are preserved and the
      output residuals array has the same shape as the input.

    Parameters
    ----------
    pts
        Input points (N, 3). Convention: columns are (x, y, z).
    center0
        Initial center guess (cx, cy, cz) in the same units as ``pts`` *after*
        scaling Z by ``1/zscale`` during fitting.
    zscale
        Divide Z by ``zscale`` during fitting; multiply residuals by ``zscale``
        on output to restore original Z units.
    keepnan
        Preserve NaNs in the Z column and return an array with the same shape as
        input.
    return_result
        If True, return a third value ``SphereFitResult``, in addition to residuals, center.
    verbose
        Print a compact report.

    kwargs
        Passed to ``scipy.optimize.minimize``. To trigger optimization. 
        Additional key ``radius``: fix the sphere radius to this value (no radius fit).
        Can include a ``maxiter`` key (common to all solvers), overridden if explicitly passed to ``fit_sphere``. If maxiter is set to 0, the function is evaluated without optimization.

    Returns
    -------
    By default:
        residuals, center
    If return_result:
        residuals, center, SphereFitResult
        
    Note that center defines automatically the best fit radius. To know this value,
        use: ``fom, deltaR, rad_fit = sphere_error(center, pts, retall=True)``
    """
    
    fom_func=sphere_error
    
    pts_in = pts #_as_points(pts)
    if zscale == 0:
        raise ValueError("`zscale` must be non-zero.")

    # Copy & scale: x/y unchanged, z scaled for fitting
    pts_fit = pts_in / np.array([1.0, 1.0, float(zscale)], dtype=float)

    # Filter NaNs once (more efficient than inside objective)
    mask = ~np.isnan(pts_fit[:, 2])
    pts_fit_valid = pts_fit[mask, :]

    c0 = np.asarray(center0, dtype=float).reshape(3)

    # Parse options dict similar to fit_cylinder.py
    fixed_radius = None
    opt_options = None
    if "options" in kwargs and isinstance(kwargs["options"], dict):
        opt_options = dict(kwargs["options"])  # copy
        if maxiter is None:
            maxiter = opt_options.pop("maxiter", None)
        fixed_radius = opt_options.pop("radius", None)
        kwargs["options"] = opt_options

    optimizer_result = None
    center = c0

    def _objective(cvec: np.ndarray) -> float:
        return sphere_error(cvec, pts_fit_valid, radius=fixed_radius, retall=False)

    if maxiter is not None and maxiter != 0:
        # If user provided a maxiter, keep it (even though we popped it) by restoring into scipy options.
        # This keeps behavior close to what people expect when passing options={'maxiter': N}.
        if opt_options is None:
            kwargs["options"] = {"maxiter": int(maxiter)}
        else:
            kwargs["options"] = {**opt_options, "maxiter": int(maxiter)}

        optimizer_result = minimize(_objective, x0=center, **kwargs)
        center = np.asarray(optimizer_result.x, dtype=float)

    fom, deltaR, rad_fit = sphere_error(
        center, pts_fit_valid, radius=fixed_radius, retall=True
    )

    # Scale residuals back to original z units
    deltaR[:, 2] *= float(zscale)
    fom_out = float(fom * float(zscale))

    if keepnan:
        out = pts_in.copy()
        out[mask, 2] = deltaR[:, 2]
        deltaR = out
    else:
        deltaR = deltaR

    if verbose:
        s = span(pts_fit, axis=0)
        print("-----------------------------------")
        print('Results of fit with function: "sphere_error" on region:')
        print("X: [%6.3f,%6.3f]" % tuple(s[0]))
        print("Y: [%6.3f,%6.3f]" % tuple(s[1]))
        print("data range (Z_fit): [%6.3f,%6.3f]" % tuple(s[2]))
        print("---")
        if optimizer_result is not None:
            print(optimizer_result)
        print("Center (fit space):", center)
        print("Radius (fit space):", rad_fit)
        print("---")
        print(f"rms residuals={fom_out} [units of original Z]")
        print("-----------------------------------")

    if return_result:
        return deltaR, center, optimizer_result
    
    return deltaR, center
