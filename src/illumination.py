"""Local terrain, horizon masks and illumination maps for a lunar landing patch.

Pure NumPy / SciPy. Conventions: azimuth 0° = North, clockwise (90° = East);
patch axes are x = East (m), y = North (m); arrays are indexed [y, x].
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

PATCH_HALF_M = 1000.0      # patch is 2 km x 2 km around the target
GRID_N = 97                # odd, so one cell sits exactly on the target
N_AZ = 48                  # horizon azimuth bins (7.5°)
SOLAR_CONSTANT = 1361.0    # W/m² at 1 AU
PANEL_EFFICIENCY = 0.30


# ----------------------------------------------------------------------------
# Terrain
# ----------------------------------------------------------------------------
def synthetic_dem(lat: float, lon: float, n: int = GRID_N, half: float = PATCH_HALF_M):
    """Demo relief (same large-scale shape as the original app) plus small craters."""
    g = np.linspace(-half, half, n)
    X, Y = np.meshgrid(g, g)
    z = (120.0 * np.exp(-((X + 180) ** 2 + (Y - 120) ** 2) / (2 * 380 ** 2))
         - 160.0 * np.exp(-((X - 260) ** 2 + (Y + 220) ** 2) / (2 * 320 ** 2))
         + 25.0 * np.sin(X / 180.0) * np.cos(Y / 180.0))
    rng = np.random.default_rng(int(abs(lat * 1000) + abs(lon * 100)))
    for _ in range(14):
        cx, cy = rng.uniform(-0.9 * half, 0.9 * half, 2)
        R, d = rng.uniform(40, 130), rng.uniform(6, 22)
        r = np.hypot(X - cx, Y - cy)
        z += np.where(r < R, -d * (1 - (r / R) ** 2), 0.0) + 0.18 * d * np.exp(-(((r - R) / (0.3 * R)) ** 2))
    return g, g.copy(), z


def load_dem_patch(path: str, lat: float, lon: float, n: int = GRID_N, half: float = PATCH_HALF_M):
    """Sample an (n x n) East/North patch from a polar-stereographic GeoTIFF.

    Returns (gx, gy, z) with z relative to the target, or None if the file,
    rasterio/pyproj or the data are unavailable (the caller falls back to the demo relief).
    """
    try:
        import rasterio
        from pyproj import Transformer

        if not os.path.exists(path):
            return None
        with rasterio.open(path) as ds:
            tr = Transformer.from_crs("+proj=longlat +R=1737400 +no_defs", ds.crs, always_xy=True)
            p0 = np.array(tr.transform(lon, lat))
            pe = np.array(tr.transform(lon + 0.05, lat))
            pn = np.array(tr.transform(lon, lat + 0.001))
            ue = (pe - p0) / np.linalg.norm(pe - p0)      # local East in map coordinates
            un = (pn - p0) / np.linalg.norm(pn - p0)      # local North in map coordinates
            g = np.linspace(-half, half, n)
            E, N = np.meshgrid(g, g)
            xs = p0[0] + E * ue[0] + N * un[0]
            ys = p0[1] + E * ue[1] + N * un[1]
            vals = np.array([v[0] for v in ds.sample(zip(xs.ravel(), ys.ravel()))], dtype=float).reshape(n, n)
            if ds.nodata is not None:
                vals[vals == ds.nodata] = np.nan
        if not np.isfinite(vals).all():
            return None
        c = n // 2
        return g, g.copy(), vals - vals[c - 1:c + 2, c - 1:c + 2].mean()
    except Exception:
        return None


def horizon_bins(gx, gy, z, n_az: int = N_AZ, n_steps: int = 96) -> np.ndarray:
    """Terrain horizon elevation (deg) seen from every cell. Shape (n_az, ny, nx).

    Ray-marches the DEM along each azimuth; the floor is 0° (geometric horizon).
    Only terrain inside the patch is considered.
    """
    ny, nx = z.shape
    dx = gx[1] - gx[0]
    jj, ii = np.mgrid[0:ny, 0:nx].astype(float)
    dists = np.linspace(dx, gx[-1] - gx[0], n_steps)
    out = np.empty((n_az, ny, nx))
    for k in range(n_az):
        a = np.radians(k * 360.0 / n_az)
        s, c = np.sin(a), np.cos(a)
        best = np.zeros((ny, nx))
        for d in dists:
            zi = map_coordinates(z, [jj + c * d / dx, ii + s * d / dx], order=1, mode="nearest")
            best = np.maximum(best, (zi - z) / d)
        out[k] = np.degrees(np.arctan(best))
    return out


def horizon_at(H: np.ndarray, az_deg) -> np.ndarray:
    """Interpolate horizon bins H (n_az, ...) at azimuths az_deg (m,) -> (m, ...)."""
    nb = H.shape[0]
    f = (np.atleast_1d(np.asarray(az_deg, float)) % 360.0) / 360.0 * nb
    i0 = np.floor(f).astype(int)
    w = (f - i0).reshape((-1,) + (1,) * (H.ndim - 1))
    return (1 - w) * H[i0 % nb] + w * H[(i0 + 1) % nb]


# ----------------------------------------------------------------------------
# Illumination
# ----------------------------------------------------------------------------
def pv_power(sun_el, visible, tracking: bool = True) -> np.ndarray:
    """PV flux (W/m²). Vertical sun-tracking arrays see the Sun at normal incidence;
    horizontal arrays follow the original app's cosine law (floored at 0.15)."""
    sun_el = np.asarray(sun_el, float)
    f = np.ones_like(sun_el) if tracking else np.clip(np.sin(np.radians(sun_el)), 0.15, 1.0)
    return np.where(visible, SOLAR_CONSTANT * PANEL_EFFICIENCY * f, 0.0)


def lit_fraction(H, az, el, mask=None) -> np.ndarray:
    """Fraction of the timeline (0..1) each cell sees a body above its horizon.
    `mask` (T,) restricts the count to selected times; the denominator stays T."""
    az, el = np.asarray(az, float), np.asarray(el, float)
    if mask is not None:
        az, el = az[mask], el[mask]
    acc = np.zeros(H.shape[1:])
    for s in range(0, len(az), 64):
        acc += (el[s:s + 64, None, None] > horizon_at(H, az[s:s + 64])).sum(0)
    return acc / (len(mask) if mask is not None else max(len(az), 1))


def margin_transect(H, az, el, axis: int = 0) -> np.ndarray:
    """Elevation of the body above the local horizon along a transect through the
    target: shape (T, nx). axis 0 = West→East, 1 = South→North."""
    c = H.shape[1] // 2
    Ht = H[:, c, :] if axis == 0 else H[:, :, c]
    return np.asarray(el, float)[:, None] - horizon_at(Ht, az)


def shade_map(gx, gy, z, H, az: float, el: float, gamma: float = 0.6) -> np.ndarray:
    """Hillshade with cast shadows for a Sun at (az, el). Returns 0..1 (0.06 = earthshine floor)."""
    dzdy, dzdx = np.gradient(z, gy, gx)
    nrm = np.sqrt(dzdx ** 2 + dzdy ** 2 + 1.0)
    a, e = np.radians(az), np.radians(el)
    ndl = (-dzdx * np.cos(e) * np.sin(a) - dzdy * np.cos(e) * np.cos(a) + np.sin(e)) / nrm
    lit = el > horizon_at(H, [az])[0]
    ndl = np.where(lit, np.clip(ndl, 0.0, None), 0.0)
    ref = np.percentile(ndl[lit], 99) if lit.any() else 1.0
    return 0.06 + 0.94 * np.clip(ndl / max(ref, 1e-6), 0.0, 1.0) ** gamma


def longest_run(flags, step_h: float) -> float:
    """Longest consecutive run of True, in hours."""
    f = np.concatenate(([0], np.asarray(flags, bool).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(f))
    return float((edges[1::2] - edges[::2]).max() * step_h) if len(edges) else 0.0


# ----------------------------------------------------------------------------
# Landing dispersion
# ----------------------------------------------------------------------------
def sample_grid(gx, gy, F, x, y) -> np.ndarray:
    """Bilinear sample of F[y, x] at metric coordinates (x, y)."""
    ix = (np.asarray(x, float) - gx[0]) / (gx[1] - gx[0])
    iy = (np.asarray(y, float) - gy[0]) / (gy[1] - gy[0])
    return map_coordinates(F, [iy, ix], order=1, mode="nearest")


def monte_carlo(gx, gy, uptime, target, sx, sy, rho, n, seed: int = 42) -> dict:
    """(X, Y) ~ N(0, Σ); a sample succeeds if the local solar uptime meets `target` (%)."""
    cov = np.array([[sx ** 2, rho * sx * sy], [rho * sx * sy, sy ** 2]])
    s = np.random.default_rng(seed).multivariate_normal([0.0, 0.0], cov, n)
    u = sample_grid(gx, gy, uptime, s[:, 0], s[:, 1])
    return {"samples": s, "uptime": u, "ok": u >= target, "cov": cov}


def ellipse_xy(cov, k: float, m: int = 121):
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, m)
    xy = vecs @ (k * np.sqrt(vals)[:, None] * np.vstack([np.cos(t), np.sin(t)]))
    return xy[0], xy[1]


def success_response(gx, gy, ok_mask, rho, sig_x, sig_y) -> np.ndarray:
    """P(success) in % for every (σy, σx) pair, integrating the Gaussian over the patch."""
    X, Y = np.meshgrid(gx, gy)
    P = np.empty((len(sig_y), len(sig_x)))
    for i, sy in enumerate(sig_y):
        for j, sx in enumerate(sig_x):
            q = (X / sx) ** 2 - 2 * rho * X * Y / (sx * sy) + (Y / sy) ** 2
            w = np.exp(-q / (2 * (1 - rho ** 2)))
            P[i, j] = 100.0 * (w * ok_mask).sum() / w.sum()
    return P


# ----------------------------------------------------------------------------
# Fallback ephemeris (same schema as EphemerisEngine.compute)
# ----------------------------------------------------------------------------
def fallback_ephemeris(lat, lon, start, days, step_h) -> pd.DataFrame:
    n = int(days * 24 / step_h) + 1
    ts = pd.date_range(pd.Timestamp(start).tz_localize("UTC"), periods=n, freq=pd.Timedelta(hours=step_h))
    t = np.arange(n) * step_h
    syn, anom = 29.530588 * 24.0, 27.55455 * 24.0
    return pd.DataFrame({
        "timestamp": ts,
        "sun_az": (np.linspace(0, 360 * days / 29.53, n) + lon) % 360,
        "sun_el": (90.0 + lat) + 1.54 * np.sin(2 * np.pi * t / syn) + 0.4 * np.cos(4 * np.pi * t / syn),
        "earth_az": (180.0 - lon + 6.5 * np.sin(2 * np.pi * t / anom)) % 360,
        "earth_el": (-lat - 84.5) + 5.2 * np.cos(2 * np.pi * t / anom),
    })

