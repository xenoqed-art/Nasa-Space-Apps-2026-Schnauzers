import numpy as np
import pandas as pd
from skyfield.api import Loader #API of the astronomic data
from src.models import LandingSite, MissionWindow


class EphemerisEngine:
    MOON_RADIUS_KM = 1737.4

    def __init__(self, data_dir: str = "data"):
        # Save NASA data
        self.load = Loader(data_dir)
        self.ts = self.load.timescale()
        self.eph = self.load("de421.bsp")
        self.moon = self.eph["moon"]
        self.sun = self.eph["sun"]
        self.earth = self.eph["earth"]

    @staticmethod
    def _icrf_to_selenographic(vec_icrf: np.ndarray, t) -> np.ndarray:
        """Transform ICRF vectrors (R3) (3, N) to moon coordinates constants (IAU)."""
        d = t.tt - 2451545.0
        T = d / 36525.0

        # Principal arguments (IAU)
        e1 = np.radians(125.045 - 0.0529921 * d)
        e2 = np.radians(250.089 - 0.1059842 * d)
        e3 = np.radians(260.008 + 13.0120009 * d)
        e4 = np.radians(176.625 + 13.3407154 * d)
        e6 = np.radians(311.589 + 26.4057084 * d)
        e10 = np.radians(15.134 - 0.1589763 * d)

        alpha0 = np.radians(
            269.9949 + 0.0031 * T
            - 3.8787 * np.sin(e1) - 0.1204 * np.sin(e2)
            + 0.0700 * np.sin(e3) - 0.0172 * np.sin(e4) + 0.0072 * np.sin(e6)
        )
        delta0 = np.radians(
            66.5392 + 0.0130 * T
            + 1.5419 * np.cos(e1) + 0.0239 * np.cos(e2)
            - 0.0278 * np.cos(e3) + 0.0068 * np.cos(e4) - 0.0029 * np.cos(e6)
        )
        w = np.radians(
            38.3213 + 13.17635815 * d - 1.4e-12 * (d ** 2)
            + 3.5610 * np.sin(e1) + 0.1208 * np.sin(e2)
            - 0.0642 * np.sin(e3) + 0.0158 * np.sin(e4)
            + 0.0252 * np.sin(e6) - 0.0066 * np.sin(e10)
        )

        a = alpha0 + np.pi / 2.0
        b = np.pi / 2.0 - delta0
        x, y, z = vec_icrf[0], vec_icrf[1], vec_icrf[2]

        # Rot R_z(w) * R_x(pi/2 - delta0) * R_z(alpha0 + pi/2)
        x1 = np.cos(a) * x + np.sin(a) * y
        y1 = -np.sin(a) * x + np.cos(a) * y
        z1 = z

        x2 = x1
        y2 = np.cos(b) * y1 + np.sin(b) * z1
        z2 = -np.sin(b) * y1 + np.cos(b) * z1

        xb = np.cos(w) * x2 + np.sin(w) * y2
        yb = -np.sin(w) * x2 + np.cos(w) * y2
        zb = z2

        return np.vstack((xb, yb, zb))

    def _to_topocentric_az_el(
        self, body_vec_seleno: np.ndarray, lat: float, lon: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate acimut y elevación (en grados) desde un punto (lat, lon) lunar."""
        phi = np.radians(lat)
        lam = np.radians(lon)

# Position of the site in the surface (including topocentic "paralaje")
        obs_x = self.MOON_RADIUS_KM * np.cos(phi) * np.cos(lam)
        obs_y = self.MOON_RADIUS_KM * np.cos(phi) * np.sin(lam)
        obs_z = self.MOON_RADIUS_KM * np.sin(phi)

        dx = body_vec_seleno[0] - obs_x
        dy = body_vec_seleno[1] - obs_y
        dz = body_vec_seleno[2] - obs_z

        # Proyecction Est-Nort-Up (ENU)
        east = -np.sin(lam) * dx + np.cos(lam) * dy
        north = (
            -np.sin(phi) * np.cos(lam) * dx
            - np.sin(phi) * np.sin(lam) * dy
            + np.cos(phi) * dz
        )
        up = (
            np.cos(phi) * np.cos(lam) * dx
            + np.cos(phi) * np.sin(lam) * dy
            + np.sin(phi) * dz
        )

        dist = np.sqrt(east**2 + north**2 + up**2)
        el_deg = np.degrees(np.arcsin(np.clip(up / dist, -1.0, 1.0)))
        az_deg = np.degrees(np.arctan2(east, north)) % 360.0
        return az_deg, el_deg

    def compute(self, site: LandingSite, window: MissionWindow) -> pd.DataFrame:
        timestamps = pd.date_range(
            start=window.start_date,
            end=window.end_date,
            freq=f"{window.step_hours}h",
        )
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize("UTC")
        else:
            timestamps = timestamps.tz_convert("UTC")

        t = self.ts.from_datetimes(timestamps.to_pydatetime())

        sun_icrf = (self.sun - self.moon).at(t).position.km
        earth_icrf = (self.earth - self.moon).at(t).position.km

        sun_seleno = self._icrf_to_selenographic(sun_icrf, t)
        earth_seleno = self._icrf_to_selenographic(earth_icrf, t)

        sun_az, sun_el = self._to_topocentric_az_el(sun_seleno, site.lat, site.lon)
        earth_az, earth_el = self._to_topocentric_az_el(earth_seleno, site.lat, site.lon)

        return pd.DataFrame({
            "timestamp": timestamps, #date and hour
            "sun_az": sun_az, #Sun Azimut
            "sun_el": sun_el, # Sun elevation
            "earth_az": earth_az, #Earth azimuth
            "earth_el": earth_el, #E azimuth
        })
