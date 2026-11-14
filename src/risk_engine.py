# Metadata: Module structure and docstrings assisted by LLM (SI).
"""
Monte Carlo Entry, Descent, and Landing (EDL) Risk & Viability Engine.

Evaluates landing site robustness under spatial touchdown dispersion at the
Lunar South Pole. Quantifies illumination uptime, Direct-to-Earth (DTE) RF
link availability, and worst-case continuous eclipse duration for thermal and
power subsystem survival.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ephemeris import EphemerisEngine
from src.models import LandingSite, MissionWindow, RiskReport

# Graceful degradation: allows full end-to-end execution using a synthetic
# terrain model while LRO LOLA Digital Elevation Model (DEM) ingestion is finalized.
try:
    from src.terrain import LunarTerrain, PowerCommsEvaluator
    HAS_TERRAIN_MODULE = True
except (ImportError, AttributeError):
    HAS_TERRAIN_MODULE = False


class MonteCarloRiskAnalyzer:
    """
    Stochastic risk assessment engine for Lunar South Pole candidate sites.

    In polar lunar operations, grazing solar incidence (~1.54 deg obliquity)
    means a 30-to-100 meter touchdown error can place a lander behind a crater
    rim, turning a near-eternal light peak into a permanent shadow trap.
    This analyzer propagates spatial landing uncertainty through local terrain
    horizon masks to produce mission-level survival probabilities.
    """

    # IAU mean lunar radius in meters (used for local tangent-plane projections)
    LUNAR_RADIUS_M: float = 1_737_400.0

    # AM0 Solar irradiance at 1 AU (W/m^2) and nominal spacecraft GaAs array efficiency
    SOLAR_CONSTANT_WM2: float = 1361.0
    NOMINAL_PANEL_EFFICIENCY: float = 0.28

    def __init__(
        self,
        ephemeris_engine: EphemerisEngine,
        min_energy_uptime: float = 0.60,
        min_comms_uptime: float = 0.50,
    ) -> None:
        """
        Initialize the risk analyzer with mission subsystem thresholds.

        Args:
            ephemeris_engine: Validated JPL celestial ephemeris provider.
            min_energy_uptime: Minimum fraction of mission duration requiring
                direct solar illumination to maintain positive power margin.
            min_comms_uptime: Minimum fraction of mission duration requiring
                unobstructed Direct-to-Earth (DTE) line-of-sight.
        """
        self.ephemeris_engine = ephemeris_engine
        self.min_energy_uptime = min_energy_uptime
        self.min_comms_uptime = min_comms_uptime

        if HAS_TERRAIN_MODULE:
            self.terrain = LunarTerrain()
            self.evaluator = PowerCommsEvaluator(self.terrain)
        else:
            self.terrain = None
            self.evaluator = None

    def _sample_landing_ellipse(
        self, site: LandingSite, n_samples: int, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        STEP 1: Sample perturbed touchdown coordinates within the EDL dispersion ellipse.

        Models touchdown error as an isotropic 2D Gaussian distribution in the
        local East-North tangent plane (dx, dy ~ N(0, sigma^2)) and projects
        metric offsets onto selenographic latitude and longitude.
        """
        rng = np.random.default_rng(seed)

        # Draw local metric offsets (East = dx, North = dy) in meters
        dx_m, dy_m = rng.normal(loc=0.0, scale=site.dispersion_m, size=(2, n_samples))

        # Convert meridional offset (dy) to degrees of selenographic latitude
        delta_lat_deg = np.degrees(dy_m / self.LUNAR_RADIUS_M)

        # Convert zonal offset (dx) to degrees of longitude, accounting for
        # meridian convergence near the lunar pole (guarded against pole singularity)
        cos_lat = max(float(np.cos(np.radians(site.lat))), 1e-4)
        delta_lon_deg = np.degrees(dx_m / (self.LUNAR_RADIUS_M * cos_lat))

        sampled_lats = np.clip(site.lat + delta_lat_deg, -90.0, 90.0)
        sampled_lons = ((site.lon + delta_lon_deg + 180.0) % 360.0) - 180.0

        return sampled_lats, sampled_lons

    @staticmethod
    def _compute_max_eclipse_hours(
        sun_visible: pd.Series, step_hours: float
    ) -> float:
        """
        STEP 2: Compute worst-case continuous solar blackout duration (in hours).

        For lunar landers, average illumination is insufficient; a single
        uninterrupted eclipse exceeding battery capacity and radioisotope/heater
        thermal limits causes mission loss. Uses vectorized run-length encoding.
        """
        in_shadow = ~sun_visible.to_numpy(dtype=bool)
        if not np.any(in_shadow):
            return 0.0

        # Pad boundaries with False to detect exact rising/falling shadow edges
        padded = np.concatenate(([False], in_shadow, [False]))
        eclipse_starts = np.where(padded[1:] & ~padded[:-1])[0]
        eclipse_ends = np.where(~padded[1:] & padded[:-1])[0]

        max_consecutive_steps = int(np.max(eclipse_ends - eclipse_starts))
        return float(max_consecutive_steps * step_hours)

    @staticmethod
    def _synthetic_polar_horizon(lat: float, lon: float) -> np.ndarray:
        """
        Fallback 360-degree topographic horizon mask (degrees elevation).
        Simulates undulating crater rim geometry when LOLA DEM data is offline.
        """
        azimuths = np.arange(360, dtype=float)
        spatial_phase = np.radians(lat * 12.0 + lon * 1.5)
        return (
            0.65
            + 1.05 * np.sin(np.radians(2.0 * azimuths) + spatial_phase)
            + 0.40 * np.cos(np.radians(5.0 * azimuths) - spatial_phase)
        )

    def evaluate_nominal_site(
        self, site: LandingSite, ephem_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        STEP 3: Perform deterministic line-of-sight and power evaluation at a single point.

        Because the Moon is an airless body (zero atmospheric refraction), optical
        and RF visibility reduce strictly to geometric elevation comparison
        against the 360-degree local topographic horizon profile.
        """
        if HAS_TERRAIN_MODULE and self.terrain is not None and self.evaluator is not None:
            horizon_profile = self.terrain.get_horizon_profile(site.lat, site.lon)
            evaluated_df = self.evaluator.evaluate(site, ephem_df)
            return evaluated_df, horizon_profile

        # Fallback geometric occlusion model
        df = ephem_df.copy()
        horizon_profile = self._synthetic_polar_horizon(site.lat, site.lon)

        sun_az_idx = np.floor(df["sun_az"].to_numpy()).astype(int) % 360
        earth_az_idx = np.floor(df["earth_az"].to_numpy()).astype(int) % 360

        terrain_h_sun = horizon_profile[sun_az_idx]
        terrain_h_earth = horizon_profile[earth_az_idx]

        df["sun_visible"] = df["sun_el"] > terrain_h_sun
        df["earth_visible"] = df["earth_el"] > terrain_h_earth

        # Instantaneous photovoltaic power density (W/m^2) on horizontal array
        solar_incidence = np.sin(np.radians(np.clip(df["sun_el"], 0.0, 90.0)))
        df["power_wm2"] = np.where(
            df["sun_visible"],
            self.SOLAR_CONSTANT_WM2 * self.NOMINAL_PANEL_EFFICIENCY * solar_incidence,
            0.0,
        )
        return df, horizon_profile

    def run(
        self, site: LandingSite, window: MissionWindow, n_samples: int = 200
    ) -> RiskReport:
        """
        STEP 4: Execute full Monte Carlo dispersion campaign and aggregate risk KPIs.

        Engineering Optimization:
            Across a ~100 m landing dispersion ellipse, topocentric celestial
            parallax shifts Sun/Earth elevation by < 0.003 deg (< 11 arcsec),
            whereas local terrain horizon masks shift by several degrees.
            Therefore, JPL ephemerides are computed once at the nominal target
            and reused across all N terrain ray-casting trials.
        """
        # 1. Compute reference JPL celestial ephemerides at nominal target
        nominal_ephem_df = self.ephemeris_engine.compute(site, window)

        # 2. Generate N perturbed touchdown coordinates within landing ellipse
        sampled_lats, sampled_lons = self._sample_landing_ellipse(site, n_samples)

        # 3. Propagate each Monte Carlo trial through local topographic masking
        trial_records: list[dict[str, float]] = []
        for trial_idx, (lat_i, lon_i) in enumerate(zip(sampled_lats, sampled_lons)):
            trial_site = LandingSite(
                name=f"{site.name}_MC_{trial_idx}",
                lat=float(lat_i),
                lon=float(lon_i),
                dispersion_m=0.0,
            )
            sim_df, _ = self.evaluate_nominal_site(trial_site, nominal_ephem_df)

            trial_records.append({
                "lat": float(lat_i),
                "lon": float(lon_i),
                "solar_uptime": float(sim_df["sun_visible"].mean()),
                "comms_uptime": float(sim_df["earth_visible"].mean()),
                "max_blackout_h": self._compute_max_eclipse_hours(
                    sim_df["sun_visible"], window.step_hours
                ),
                "mean_power_wm2": float(sim_df["power_wm2"].mean()),
            })

        mc_df = pd.DataFrame(trial_records)

        # 4. Compute empirical mission success probabilities & tail risk metrics
        prob_energy = float((mc_df["solar_uptime"] >= self.min_energy_uptime).mean())
        prob_comms = float((mc_df["comms_uptime"] >= self.min_comms_uptime).mean())
        expected_solar = float(mc_df["solar_uptime"].mean())
        expected_comms = float(mc_df["comms_uptime"].mean())

        # 95th percentile worst-case continuous darkness (standard flight-rule margin)
        p95_blackout_h = float(np.percentile(mc_df["max_blackout_h"], 95))

        # 5. Compute composite Mission Viability Score (0 to 100 scale)
        # Weighted toward hard subsystem survival constraints (power & DTE link)
        viability_score = 100.0 * (
            0.45 * prob_energy + 0.35 * prob_comms + 0.20 * expected_solar
        )

        return RiskReport(
            site=site,
            prob_energy_success=prob_energy,
            prob_comms_success=prob_comms,
            expected_solar_uptime=expected_solar,
            expected_comms_uptime=expected_comms,
            worst_blackout_hours=p95_blackout_h,
            viability_score=round(viability_score, 1),
            mc_samples_df=mc_df,
        )
