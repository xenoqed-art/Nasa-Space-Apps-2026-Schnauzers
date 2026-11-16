"""Lunar Landing Site Navigator: Streamlit dashboard (Team Schnauzers, NASA Space Apps 2026)."""
import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

from src import illumination as ill
from src import visuals as vz

DEM_PATH = "data/mosaic-sfs_v2.tif"
CONFIDENCE = 90.0   # % confidence drawn as a level surface in the precision chart

CANDIDATE_SITES = {
    "Shackleton Crater Rim": (-89.54, 0.00),
    "Connecting Ridge (Shackleton–de Gerlache)": (-89.45, -135.00),
    "Malapert Massif Peak": (-86.00, 0.00),
    "Nobile Rim 1": (-85.20, 55.00),
    "Custom Coordinates": None,
}

st.set_page_config(page_title="Lunar Landing Site Navigator", page_icon="🌑", layout="wide", initial_sidebar_state="expanded")
st.markdown(vz.CSS, unsafe_allow_html=True)
fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", lambda f: f)


def show(fn, *args, **kwargs):
    """Stretch a Streamlit element to the container width on old and new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def plot(fig):
    show(st.plotly_chart, fig, theme=None, config=vz.PLOT_CONFIG)


# -----------------------------------------------------------------------------
# Cached computation
# -----------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    from src.ephemeris import EphemerisEngine
    return EphemerisEngine("data")


@st.cache_data(show_spinner="Computing Sun and Earth positions…")
def load_ephemeris(lat, lon, start, days, step_h):
    try:
        t0 = pd.Timestamp(start).tz_localize("UTC")
        window = SimpleNamespace(start_date=t0, end_date=t0 + pd.Timedelta(days=days), step_hours=step_h)
        return get_engine().compute(SimpleNamespace(lat=lat, lon=lon), window), "JPL DE421 via Skyfield", True
    except Exception as exc:   # shown in the header, never silent
        return ill.fallback_ephemeris(lat, lon, start, days, step_h), f"demo model ({type(exc).__name__})", False


@st.cache_data(show_spinner="Reading terrain and tracing horizons…")
def load_terrain(lat, lon):
    patch = ill.load_dem_patch(DEM_PATH, lat, lon)
    real = patch is not None
    gx, gy, z = patch if real else ill.synthetic_dem(lat, lon)
    return {"gx": gx, "gy": gy, "z": z, "H": ill.horizon_bins(gx, gy, z)}, ("GeoTIFF DEM" if real else "demo relief"), real


@st.cache_data(show_spinner="Mapping illumination…")
def simulate(lat, lon, start, days, step_h, min_power, tracking):
    terr, dem_src, dem_real = load_terrain(lat, lon)
    eph, eph_src, eph_real = load_ephemeris(lat, lon, start, days, step_h)
    H = terr["H"]
    c = H.shape[1] // 2
    Hc = H[:, c, c]                                           # horizon seen from the nominal target
    az_s, el_s = eph["sun_az"].to_numpy(), eph["sun_el"].to_numpy()
    az_e, el_e = eph["earth_az"].to_numpy(), eph["earth_el"].to_numpy()
    sun_vis = el_s > ill.horizon_at(Hc, az_s)
    earth_vis = el_e > ill.horizon_at(Hc, az_e)
    pv = ill.pv_power(el_s, sun_vis, tracking)
    powered = ill.pv_power(el_s, True, tracking) >= min_power  # times when an unobstructed array reaches the threshold
    haz = np.arange(0.0, 361.0, 2.0)
    tel = pd.DataFrame({
        "Timestamp_UTC": eph["timestamp"], "Elapsed_Hours": np.arange(len(eph)) * step_h,
        "Sun_Azimuth_deg": az_s.round(3), "Sun_Elevation_deg": el_s.round(3),
        "Earth_Azimuth_deg": az_e.round(3), "Earth_Elevation_deg": el_e.round(3),
        "Sun_Visible": sun_vis, "DTE_Visible": earth_vis, "PV_Power_Wm2": pv.round(2),
    })
    return {
        "terr": terr, "ts": eph["timestamp"], "days": np.arange(len(eph)) * step_h / 24.0, "tel": tel,
        "sun": {"az": az_s, "el": el_s, "vis": sun_vis}, "earth": {"az": az_e, "el": el_e, "vis": earth_vis},
        "pv": pv, "haz": haz, "hor": ill.horizon_at(Hc, haz),
        "up": 100.0 * ill.lit_fraction(H, az_s, el_s, powered),
        "eph_src": eph_src, "eph_real": eph_real, "dem_src": dem_src, "dem_real": dem_real,
    }


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Mission configuration")
    site_choice = st.selectbox("Candidate landing site", list(CANDIDATE_SITES))
    default_lat, default_lon = CANDIDATE_SITES[site_choice] or (-89.50, 0.00)
    c_lat, c_lon = st.columns(2)
    lat = c_lat.number_input("Latitude (°)", min_value=-90.0, max_value=-75.0, value=default_lat, step=0.05, format="%.2f")
    lon = c_lon.number_input("Longitude (°)", min_value=-180.0, max_value=180.0, value=default_lon, step=0.50, format="%.2f")

    st.divider()
    st.subheader("Timeline")
    start_date = st.date_input("Landing epoch (UTC)", value=datetime.date(2028, 10, 15))
    duration_days = st.slider("Mission duration (Earth days)", min_value=7, max_value=90, value=30, step=1)
    step_hours = st.select_slider("Ephemeris step (hours)", options=[0.5, 1.0, 2.0, 3.0, 6.0], value=1.0)

    st.divider()
    st.subheader("Power system")
    array_mode = st.radio("Solar array", ["Vertical, sun-tracking", "Horizontal, fixed"],
                          help="Near the pole the Sun skims the horizon. A vertical array faces it head-on; a flat one barely collects.")
    min_power = st.number_input("Minimum PV power (W/m²)", min_value=50.0, max_value=400.0, value=150.0, step=10.0)
    target_uptime = st.slider("Required solar uptime (%)", min_value=10, max_value=100, value=70, step=5)

    st.divider()
    st.subheader("Landing dispersion")
    sigma_x = st.slider("Cross-track σx (m)", min_value=25, max_value=400, value=120, step=5)
    sigma_y = st.slider("Down-track σy (m)", min_value=25, max_value=400, value=180, step=5)
    rho = st.slider("Correlation ρ", min_value=-0.90, max_value=0.90, value=0.25, step=0.05)
    n_samples = st.select_slider("Monte Carlo samples", options=[500, 1000, 2500, 5000], value=1000)

# -----------------------------------------------------------------------------
# Simulation and readout
# -----------------------------------------------------------------------------
S = simulate(lat, lon, start_date, duration_days, step_hours, float(min_power), array_mode.startswith("Vertical"))
gx, gy, z, H = S["terr"]["gx"], S["terr"]["gy"], S["terr"]["z"], S["terr"]["H"]
mc = ill.monte_carlo(gx, gy, S["up"], target_uptime, float(sigma_x), float(sigma_y), rho, n_samples)

powered = S["pv"] >= min_power
st.markdown(vz.hero_html(S["eph_src"], S["eph_real"], S["dem_src"], S["dem_real"]), unsafe_allow_html=True)
st.markdown(vz.readout_html([
    ("Mean PV power", f"{S['pv'].mean():.0f}", "W/m²", f"threshold {min_power:.0f} W/m²"),
    ("Solar uptime", f"{powered.mean() * 100:.1f}", "%", "at the nominal target"),
    ("Direct-to-Earth", f"{S['earth']['vis'].mean() * 100:.1f}", "%", "line of sight to Earth"),
    ("Longest blackout", f"{ill.longest_run(~powered, step_hours):.0f}", "h", "without solar power"),
    ("Landing success", f"{mc['ok'].mean() * 100:.1f}", "%", f"{n_samples:,} runs, uptime ≥ {target_uptime}%"),
]), unsafe_allow_html=True)

tab_margin, tab_sky, tab_risk, tab_data = st.tabs(["Illumination and Earth link", "Terrain and sky", "Landing risk", "Telemetry"])

# --- Tab 1: space-time margin surface ---------------------------------------
with tab_margin:
    c1, c2 = st.columns(2)
    body = c1.radio("Body", ["Sun (solar power)", "Earth (direct-to-Earth link)"], horizontal=True)
    axis = c2.radio("Transect through the target", ["West → East", "South → North"], horizontal=True)
    kind, west_east = ("sun" if body.startswith("Sun") else "earth"), axis.startswith("West")
    b = S[kind]
    M = ill.margin_transect(H, b["az"], b["el"], axis=0 if west_east else 1)
    plot(vz.fig_spacetime(gx if west_east else gy, S["days"], M, kind, "West → East (m)" if west_east else "South → North (m)"))
    st.caption("Height is how far the body sits above the terrain horizon, seen from each point along the transect, "
               "day after day. Below the white 0° contour the point is in shadow, or has no line of sight to Earth.")

# --- Tab 2: lit terrain + sky panorama, driven by an epoch cursor -----------
@fragment
def sky_and_terrain():
    c1, c2 = st.columns([3, 1])
    day = c1.slider("Epoch cursor (mission day)", 0.0, float(duration_days), 0.0, step=step_hours / 24.0, format="%.1f")
    vexag = c2.select_slider("Vertical exaggeration", options=[1, 2, 3, 5], value=2, format_func=lambda v: f"×{v}")
    idx = int(min(round(day * 24.0 / step_hours), len(S["days"]) - 1))
    sun, earth = S["sun"], S["earth"]
    shade = ill.shade_map(gx, gy, z, H, float(sun["az"][idx]), float(sun["el"][idx]))
    left, right = st.columns(2)
    with left:
        plot(vz.fig_terrain(gx, gy, z, shade, vexag))
    with right:
        plot(vz.fig_sky(S["haz"], S["hor"], sun, earth, idx))
    st.caption(f"{S['ts'].iloc[idx]:%d %b %Y, %H:%M} UTC. "
               f"Sun at azimuth {sun['az'][idx]:.1f}°, elevation {sun['el'][idx]:+.2f}° "
               f"({'above' if sun['vis'][idx] else 'hidden behind'} the local horizon). "
               f"Earth at azimuth {earth['az'][idx]:.1f}°, elevation {earth['el'][idx]:+.2f}° "
               f"({'in line of sight' if earth['vis'][idx] else 'blocked'}). Contours on the floor are terrain level curves.")


with tab_sky:
    sky_and_terrain()

# --- Tab 3: landing risk surfaces -------------------------------------------
with tab_risk:
    if S["up"].max() < target_uptime:
        st.info(f"No point in this 2 km patch reaches {target_uptime}% solar uptime for these dates. "
                "Lower the required uptime or move the landing epoch to see where the risk boundary falls.")
    left, right = st.columns([1.15, 1])
    with left:
        plot(vz.fig_uptime(gx, gy, S["up"], mc, target_uptime))
        st.caption("Height is the solar uptime of each landing point. The flat plane is your requirement; the white contour "
                   "is where the terrain crosses it. Dots are simulated landings, rings are 1σ, 2σ and 3σ.")
    with right:
        sigmas = np.linspace(25.0, 400.0, 12)
        ok_mask = S["up"] >= target_uptime
        P = ill.success_response(gx, gy, ok_mask, rho, sigmas, sigmas)
        p_now = ill.success_response(gx, gy, ok_mask, rho, [float(sigma_x)], [float(sigma_y)])[0, 0]
        plot(vz.fig_response(sigmas, sigmas, P, (float(sigma_x), float(sigma_y), float(p_now)), CONFIDENCE))
        st.caption(f"How precise the landing must be: success probability for every dispersion. Beyond the white contour the "
                   f"chance of success is above {CONFIDENCE:.0f}%. The pin marks your current σx, σy "
                   f"({p_now:.0f}%, integrated over the 2 km patch).")

# --- Tab 4: telemetry --------------------------------------------------------
with tab_data:
    show(st.dataframe, S["tel"], height=380)
    st.download_button("Download telemetry CSV", S["tel"].to_csv(index=False).encode("utf-8"),
                       file_name=f"lunar_site_lat{lat:.2f}_lon{lon:.2f}.csv", mime="text/csv")

with st.expander("Model notes"):
    st.markdown(
        "- Sun and Earth directions come from `EphemerisEngine` (JPL DE421). If it cannot run, the header says so and a demo model is used.\n"
        "- Horizons are traced over the 2 km patch only. Distant massifs, such as Malapert, are not included.\n"
        "- Solar uptime per point is the share of the timeline in which the Sun clears that point's horizon and the array reaches the power threshold.\n"
        "- A landing succeeds when its local solar uptime meets the requirement.\n"
        "- The precision chart integrates a bivariate normal over the patch, so σ beyond about 300 m is truncated by the patch edge."
    )

