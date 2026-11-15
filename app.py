import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

terrain_engine = TerrainEngine(dem_path="data/mosaic-sfs_v2.tif")
# -----------------------------------------------------------------------------
# 1. BACKEND MODULE IMPORTS (Team Schnauzers Architecture)
# -----------------------------------------------------------------------------
try:
    from src import ephemeris, terrain, models, risk_engine
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

# -----------------------------------------------------------------------------
# 2. PAGE CONFIGURATION & GLOBAL STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Lunar Landing Site Navigator",
    page_icon="🌑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark scientific theme for Matplotlib figures
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#161B22",
    "axes.edgecolor": "#30363D",
    "axes.labelcolor": "#E6EDF3",
    "text.color": "#E6EDF3",
    "xtick.color": "#8B949E",
    "ytick.color": "#8B949E",
    "grid.color": "#21262D",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})

# Candidate South Pole Sites (Lat, Lon)
CANDIDATE_SITES = {
    "Shackleton Crater Rim": (-89.54, 0.00),
    "Connecting Ridge (Shackleton–de Gerlache)": (-89.45, -135.00),
    "Malapert Massif Peak": (-86.00, 0.00),
    "Nobile Rim 1": (-85.20, 55.00),
    "Custom Coordinates": None,
}

# -----------------------------------------------------------------------------
# 3. CACHED COMPUTATION PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_mission_simulation(
    lat: float,
    lon: float,
    start_date: datetime.date,
    duration_days: int,
    step_hours: float,
    sigma_x: float,
    sigma_y: float,
    rho: float,
    n_samples: int,
    min_power_wm2: float,
) -> dict:
    """
    Orchestrates celestial ephemeris, 360-deg horizon occlusion, PV power,
    and bivariate normal Monte Carlo landing dispersion risk analysis.
    """
    # --- A. Time Grid ---
    n_steps = int((duration_days * 24) / step_hours) + 1
    timestamps = pd.date_range(
        start=pd.Timestamp(start_date),
        periods=n_steps,
        freq=pd.Timedelta(hours=step_hours),
    )
    t_hours = np.arange(n_steps) * step_hours

    # -------------------------------------------------------------------------
    # TODO: Replace the fallback physics below with your exact src/* calls
    # once all module signatures are finalized. Example mapping:
    #
    # ephem_df = ephemeris.compute_topocentric(lat, lon, timestamps)
    # horizon_az, horizon_el = terrain.get_horizon_profile(lat, lon)
    # pv_power = models.estimate_pv_power(ephem_df, horizon_el)
    # mc_results = risk_engine.run_monte_carlo(lat, lon, sigma_x, sigma_y, rho, n_samples)
    # -------------------------------------------------------------------------

    # 1. Celestial Ephemeris (Sun & Earth topocentric angles)
    synodic_period_h = 29.530588 * 24.0
    anomalistic_period_h = 27.55455 * 24.0
    axial_tilt_deg = 1.54

    sun_az = (np.linspace(0, 360 * (duration_days / 29.53), n_steps) + lon) % 360
    sun_el = (
        (90.0 + lat)
        + axial_tilt_deg * np.sin(2 * np.pi * t_hours / synodic_period_h)
        + 0.4 * np.cos(2 * np.pi * t_hours / (synodic_period_h / 2))
    )

    # Earth libration oscillation near lunar horizon
    earth_az = (180.0 - lon + 6.5 * np.sin(2 * np.pi * t_hours / anomalistic_period_h)) % 360
    earth_el = (-lat - 84.5) + 5.2 * np.cos(2 * np.pi * t_hours / anomalistic_period_h)

    # 2. 360-degree Topographic Horizon Profile & DEM Patch
    azimuth_grid = np.linspace(0, 360, 361)
    rng_terrain = np.random.default_rng(seed=int(abs(lat * 1000) + abs(lon * 100)))
    horizon_el = (
        1.2
        + 1.8 * np.sin(np.deg2rad(2 * azimuth_grid + lon))
        + 0.9 * np.cos(np.deg2rad(5 * azimuth_grid))
        + 0.3 * np.sin(np.deg2rad(11 * azimuth_grid))
    )
    horizon_el = np.clip(horizon_el, -1.0, 6.0)

    # Interpolate horizon elevation along Sun and Earth azimuth tracks
    sun_horizon_mask = np.interp(sun_az, azimuth_grid, horizon_el)
    earth_horizon_mask = np.interp(earth_az, azimuth_grid, horizon_el)

    sun_visible = sun_el > sun_horizon_mask
    dte_visible = earth_el > earth_horizon_mask

    # 3. Photovoltaic Power Estimation (W/m^2)
    solar_constant = 1361.0  # W/m^2 at 1 AU
    panel_efficiency = 0.30
    incidence_factor = np.clip(np.cos(np.deg2rad(90.0 - np.clip(sun_el, 0, 90))), 0.15, 1.0)
    pv_power = np.where(sun_visible, solar_constant * panel_efficiency * incidence_factor, 0.0)

    # 4. Synthetic DEM Grid (2km x 2km local patch around target)
    grid_lim = 1000.0  # meters
    gx = np.linspace(-grid_lim, grid_lim, 120)
    gy = np.linspace(-grid_lim, grid_lim, 120)
    GX, GY = np.meshgrid(gx, gy)
    dem_z = (
        120.0 * np.exp(-((GX + 180) ** 2 + (GY - 120) ** 2) / (2 * 380 ** 2))
        - 160.0 * np.exp(-((GX - 260) ** 2 + (GY + 220) ** 2) / (2 * 320 ** 2))
        + 25.0 * np.sin(GX / 180.0) * np.cos(GY / 180.0)
    )

    # 5. Monte Carlo Bivariate Normal Dispersion Analysis: (X, Y) ~ N(mu, Sigma)
    cov_xy = rho * sigma_x * sigma_y
    cov_matrix = np.array([[sigma_x ** 2, cov_xy], [cov_xy, sigma_y ** 2]])
    rng_mc = np.random.default_rng(seed=42)
    samples = rng_mc.multivariate_normal(mean=[0.0, 0.0], cov=cov_matrix, size=n_samples)

    # Evaluate local terrain elevation at each dispersed landing point
    sample_z = (
        120.0 * np.exp(-((samples[:, 0] + 180) ** 2 + (samples[:, 1] - 120) ** 2) / (2 * 380 ** 2))
        - 160.0 * np.exp(-((samples[:, 0] - 260) ** 2 + (samples[:, 1] + 220) ** 2) / (2 * 320 ** 2))
        + 25.0 * np.sin(samples[:, 0] / 180.0) * np.cos(samples[:, 1] / 180.0)
    )

    # Higher terrain elevation yields better solar uptime; crater floors suffer occlusion
    base_uptime = np.mean(pv_power >= min_power_wm2) * 100.0
    sample_uptime = np.clip(base_uptime + (sample_z * 0.14) + rng_mc.normal(0, 2.5, n_samples), 0.0, 100.0)
    target_uptime_req = 70.0  # Minimum required mission power uptime (%)
    sample_success = sample_uptime >= target_uptime_req

    telemetry_df = pd.DataFrame({
        "Timestamp_UTC": timestamps,
        "Elapsed_Hours": t_hours,
        "Sun_Azimuth_deg": np.round(sun_az, 3),
        "Sun_Elevation_deg": np.round(sun_el, 3),
        "Earth_Azimuth_deg": np.round(earth_az, 3),
        "Earth_Elevation_deg": np.round(earth_el, 3),
        "Sun_Visible": sun_visible,
        "DTE_Visible": dte_visible,
        "PV_Power_Wm2": np.round(pv_power, 2),
    })

    return {
        "telemetry": telemetry_df,
        "azimuth_grid": azimuth_grid,
        "horizon_el": horizon_el,
        "dem_gx": GX,
        "dem_gy": GY,
        "dem_z": dem_z,
        "mc_samples": samples,
        "mc_uptime": sample_uptime,
        "mc_success": sample_success,
        "cov_matrix": cov_matrix,
        "target_uptime_req": target_uptime_req,
    }


def add_confidence_ellipse(ax, cov_matrix: np.ndarray, n_std: float, edgecolor: str, label: str):
    """Draws a bivariate normal confidence ellipse on a 2D Matplotlib axis."""
    vals, vecs = np.linalg.eigh(cov_matrix)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = patches.Ellipse(
        xy=(0.0, 0.0),
        width=width,
        height=height,
        angle=theta,
        edgecolor=edgecolor,
        facecolor="none",
        lw=1.8,
        linestyle="--" if n_std > 1 else "-",
        label=label,
    )
    ax.add_patch(ellipse)


# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🛰️ Mission Configuration")

    site_choice = st.selectbox("Candidate Landing Site", list(CANDIDATE_SITES.keys()))
    if CANDIDATE_SITES[site_choice] is not None:
        default_lat, default_lon = CANDIDATE_SITES[site_choice]
    else:
        default_lat, default_lon = -89.50, 0.00

    col_lat, col_lon = st.columns(2)
    lat = col_lat.number_input("Latitude (°)", min_value=-90.0, max_value=-75.0, value=default_lat, step=0.05, format="%.2f")
    lon = col_lon.number_input("Longitude (°)", min_value=-180.0, max_value=180.0, value=default_lon, step=0.50, format="%.2f")

    st.divider()
    st.subheader("⏱️ Mission Timeline")
    start_date = st.date_input("Landing Epoch (UTC)", value=datetime.date(2028, 10, 15))
    duration_days = st.slider("Mission Duration (Earth Days)", min_value=7, max_value=90, value=30, step=1)
    step_hours = st.select_slider("Ephem Time Step (Hours)", options=[0.5, 1.0, 2.0, 3.0, 6.0], value=1.0)

    st.divider()
    st.subheader("🎯 Monte Carlo Dispersion")
    sigma_x = st.slider("Cross-Track Std Dev σx (m)", min_value=25, max_value=400, value=120, step=5)
    sigma_y = st.slider("Down-Track Std Dev σy (m)", min_value=25, max_value=400, value=180, step=5)
    rho = st.slider("Correlation Coefficient (ρ)", min_value=-0.90, max_value=0.90, value=0.25, step=0.05)
    n_samples = st.select_slider("Monte Carlo Samples (N)", options=[500, 1000, 2500, 5000], value=1000)

    st.divider()
    min_power_wm2 = st.number_input("Min PV Power Threshold (W/m²)", min_value=50.0, max_value=400.0, value=150.0, step=10.0)

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD HEADER & EXECUTION
# -----------------------------------------------------------------------------
st.title("🌑 Lunar Landing Site Navigator")
st.caption("South Pole Illumination, Direct-to-Earth Line-of-Sight & Probabilistic Risk Analyzer | Team Schnauzers — NASA Space Apps 2026")

results = run_mission_simulation(
    lat=lat,
    lon=lon,
    start_date=start_date,
    duration_days=duration_days,
    step_hours=step_hours,
    sigma_x=float(sigma_x),
    sigma_y=float(sigma_y),
    rho=rho,
    n_samples=n_samples,
    min_power_wm2=min_power_wm2,
)

df = results["telemetry"]

# Top KPI Summary Metrics
mean_power = df["PV_Power_Wm2"].mean()
solar_uptime = (df["PV_Power_Wm2"] >= min_power_wm2).mean() * 100.0
dte_uptime = df["DTE_Visible"].mean() * 100.0
prob_success = results["mc_success"].mean() * 100.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Mean Photovoltaic Power", f"{mean_power:.1f} W/m²", delta=f"Target ≥ {min_power_wm2:.0f} W/m²")
k2.metric("Nominal Solar Uptime", f"{solar_uptime:.1f}%")
k3.metric("Direct-to-Earth (DTE) Uptime", f"{dte_uptime:.1f}%")
k4.metric("Mission Success Probability", f"{prob_success:.1f}%", delta=f"N = {n_samples} MC runs")

st.divider()

# -----------------------------------------------------------------------------
# 6. VISUALIZATION TABS
# -----------------------------------------------------------------------------
tab_timeline, tab_horizon, tab_mc, tab_data = st.tabs([
    "☀️ Illumination & DTE Timeline",
    "🏔️ 360° Topographic Horizon",
    "🎲 Monte Carlo Landing Dispersion",
    "📊 Telemetry Export",
])

# --- TAB 1: Ephemeris & Power Timeline ---
with tab_timeline:
    fig1, (ax_el, ax_pw) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})

    ax_el.plot(df["Timestamp_UTC"], df["Sun_Elevation_deg"], color="#FDB813", lw=1.8, label="Sun Elevation (°)")
    ax_el.plot(df["Timestamp_UTC"], df["Earth_Elevation_deg"], color="#58A6FF", lw=1.6, label="Earth Elevation (°)")
    ax_el.axhline(0.0, color="#8B949E", linestyle=":", lw=1.0, label="Geometric Horizon (0°)")
    ax_el.set_ylabel("Elevation Angle (°)")
    ax_el.set_title(f"Topocentric Sun & Earth Ephemeris at ({lat:.2f}°, {lon:.2f}°)", fontsize=12, fontweight="bold")
    ax_el.legend(loc="upper right", frameon=True, facecolor="#0E1117")
    ax_el.grid(True)

    ax_pw.fill_between(df["Timestamp_UTC"], df["PV_Power_Wm2"], color="#FDB813", alpha=0.25)
    ax_pw.plot(df["Timestamp_UTC"], df["PV_Power_Wm2"], color="#FDB813", lw=1.5, label="Estimated PV Power (W/m²)")
    ax_pw.axhline(min_power_wm2, color="#F85149", linestyle="--", lw=1.2, label=f"Operational Threshold ({min_power_wm2:.0f} W/m²)")

    # Highlight DTE link availability along bottom bar
    dte_y = np.where(df["DTE_Visible"], min_power_wm2 * 0.12, np.nan)
    ax_pw.plot(df["Timestamp_UTC"], dte_y, color="#3FB950", lw=3.5, label="Direct-to-Earth (DTE) Link Active")

    ax_pw.set_ylabel("Power Flux (W/m²)")
    ax_pw.set_xlabel("Mission Timeline (UTC)")
    ax_pw.legend(loc="upper right", frameon=True, facecolor="#0E1117")
    ax_pw.grid(True)

    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

# --- TAB 2: 360° Topographic Horizon Occlusion ---
with tab_horizon:
    fig2, ax_hor = plt.subplots(figsize=(12, 5))

    az = results["azimuth_grid"]
    hor = results["horizon_el"]

    ax_hor.fill_between(az, -2.0, hor, color="#30363D", alpha=0.85, label="Local Terrain Horizon Mask")
    ax_hor.plot(az, hor, color="#8B949E", lw=1.8)

    ax_hor.scatter(df["Sun_Azimuth_deg"], df["Sun_Elevation_deg"], c="#FDB813", s=10, alpha=0.75, label="Solar Trajectory")
    ax_hor.scatter(df["Earth_Azimuth_deg"], df["Earth_Elevation_deg"], c="#58A6FF", s=10, alpha=0.75, label="Earth Libration Track")

    ax_hor.set_xlim(0, 360)
    ax_hor.set_xticks(np.arange(0, 361, 45))
    ax_hor.set_xticklabels(["0° (N)", "45° (NE)", "90° (E)", "135° (SE)", "180° (S)", "225° (SW)", "270° (W)", "315° (NW)", "360° (N)"])
    ax_hor.set_ylim(-2.0, max(8.0, df["Sun_Elevation_deg"].max() + 2.0))
    ax_hor.set_xlabel("Topocentric Azimuth (°)")
    ax_hor.set_ylabel("Elevation Angle (°)")
    ax_hor.set_title("360° Topographic Horizon Occlusion Profile", fontsize=12, fontweight="bold")
    ax_hor.legend(loc="upper right", frameon=True, facecolor="#0E1117")
    ax_hor.grid(True)

    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# --- TAB 3: Monte Carlo Landing Dispersion ---
with tab_mc:
    col_map, col_hist = st.columns([1.35, 1])

    with col_map:
        fig3, ax_mc = plt.subplots(figsize=(7.5, 6))
        contour = ax_mc.contourf(results["dem_gx"], results["dem_gy"], results["dem_z"], levels=22, cmap="cividis", alpha=0.85)
        cbar = fig3.colorbar(contour, ax=ax_mc, pad=0.02)
        cbar.set_label("Relative Elevation (m)")

        samples = results["mc_samples"]
        success = results["mc_success"]

        ax_mc.scatter(samples[success, 0], samples[success, 1], c="#3FB950", s=12, alpha=0.65, label="Safe Landing (Uptime ≥ 70%)")
        ax_mc.scatter(samples[~success, 0], samples[~success, 1], c="#F85149", s=14, alpha=0.75, label="Occluded / High Risk")
        ax_mc.scatter([0], [0], c="#FFFFFF", marker="+", s=120, lw=2, zorder=5, label="Nominal Target μ")

        add_confidence_ellipse(ax_mc, results["cov_matrix"], n_std=1.0, edgecolor="#FFFFFF", label="1σ Dispersion Ellipse")
        add_confidence_ellipse(ax_mc, results["cov_matrix"], n_std=2.0, edgecolor="#58A6FF", label="2σ Dispersion Ellipse")
        add_confidence_ellipse(ax_mc, results["cov_matrix"], n_std=3.0, edgecolor="#FDB813", label="3σ Dispersion Ellipse")

        ax_mc.set_xlim(-750, 750)
        ax_mc.set_ylim(-750, 750)
        ax_mc.set_xlabel("Cross-Track Offset ΔX (m)")
        ax_mc.set_ylabel("Down-Track Offset ΔY (m)")
        ax_mc.set_title("Bivariate Normal Landing Dispersion (X, Y) ~ N(μ, Σ)", fontsize=11, fontweight="bold")
        ax_mc.legend(loc="upper right", fontsize=8, frameon=True, facecolor="#0E1117")
        ax_mc.grid(True)

        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    with col_hist:
        fig4, ax_hist = plt.subplots(figsize=(6, 6))
        ax_hist.hist(results["mc_uptime"], bins=25, color="#58A6FF", edgecolor="#0E1117", alpha=0.85)
        ax_hist.axvline(
            results["target_uptime_req"],
            color="#F85149",
            linestyle="--",
            lw=2,
            label=f"Min Uptime Requirement ({results['target_uptime_req']:.0f}%)",
        )
        ax_hist.axvline(
            np.mean(results["mc_uptime"]),
            color="#3FB950",
            linestyle="-",
            lw=2,
            label=f"Expected Uptime ({np.mean(results['mc_uptime']):.1f}%)",
        )
        ax_hist.set_xlabel("Simulated Solar Power Uptime (%)")
        ax_hist.set_ylabel("Monte Carlo Sample Count")
        ax_hist.set_title("Empirical Solar Uptime Distribution", fontsize=11, fontweight="bold")
        ax_hist.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="#0E1117")
        ax_hist.grid(True)

        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

# --- TAB 4: Telemetry Data & CSV Download ---
with tab_data:
    st.subheader("Simulated Mission Telemetry")
    st.dataframe(df, use_container_width=True, height=360)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Telemetry CSV",
        data=csv_bytes,
        file_name=f"lunar_site_lat{lat:.2f}_lon{lon:.2f}.csv",
        mime="text/csv",
    )
