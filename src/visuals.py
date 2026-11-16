"""Visual identity and Plotly 3D level-surface figures for the Lunar Landing Site Navigator."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import illumination as ill

# ----------------------------------------------------------------------------
# Palette, taken from the Moon itself
# ----------------------------------------------------------------------------
BASALT, REGOLITH, DUST, BONE = "#101318", "#1A1E25", "#2A2F38", "#E6E2D8"   # mare, soil, shadow edge, highlands
MUTED, SUN, EARTH = "#8A909C", "#E9B872", "#6FB1E8"                         # shadow grey, sunlit rim, earthshine
SAFE, RISK = "#8FD0B0", "#D9674B"                                          # teal frost, iron oxide
FONT = "Instrument Sans, system-ui, sans-serif"

LUNAR = [[0, "#07080b"], [0.22, "#23262d"], [0.5, "#5d5e63"], [0.76, "#a9a597"], [1, "#f3eee2"]]
UPTIME = [[0, "#0a0d14"], [0.3, "#1f2f4d"], [0.6, "#6b6a74"], [0.8, "#cfa561"], [1, "#fff0c2"]]
RESPONSE = [[0, "#4a1913"], [0.45, "#c06a45"], [0.75, "#d9c9a0"], [1, "#8fd0b0"]]
FLAT = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)   # colour = data, no extra lighting
SOFT = dict(ambient=0.75, diffuse=0.6, specular=0.08, roughness=0.8)
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Sora:wght@200;300;400&display=swap');
html, body, .stApp { font-family: 'Instrument Sans', system-ui, sans-serif; }
.stApp { background: radial-gradient(1100px 520px at 88% -8%, rgba(111,177,232,.10), transparent 62%), #101318; }
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #171A20; border-right: 1px solid #262B33; }
.block-container { padding-top: 2.4rem; max-width: 1500px; }
.hero { display: flex; gap: 1.4rem; align-items: center; margin-bottom: 1.2rem; }
.hero .mark { width: 68px; height: 68px; flex: none; }
.hero-title { font: 300 2.6rem/1.05 Sora, sans-serif; letter-spacing: -.02em; color: #E6E2D8; }
.hero-sub { margin-top: .4rem; color: #8A909C; max-width: 64ch; font-size: .95rem; line-height: 1.5; }
.hero-src { margin-left: auto; text-align: right; font-size: .8rem; color: #8A909C; line-height: 1.8; }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: .45rem; background: #8FD0B0; }
.dot.warn { background: #E9B872; }
.readout { display: grid; grid-template-columns: repeat(5, 1fr); border-top: 1px solid #2A2F38; border-bottom: 1px solid #2A2F38; margin: .6rem 0 1.4rem; }
.ro { padding: .9rem 1.1rem; border-left: 1px solid #2A2F38; }
.ro:first-child { border-left: 0; padding-left: 0; }
.ro-l { color: #8A909C; font-size: .82rem; }
.ro-v { font: 300 2.15rem/1.15 Sora, sans-serif; color: #E6E2D8; font-variant-numeric: tabular-nums; }
.ro-v span { font-size: .9rem; color: #8A909C; margin-left: .3rem; }
.ro-n { color: #8A909C; font-size: .78rem; }
button[data-baseweb="tab"] { font-size: .95rem; }
button[data-baseweb="tab"][aria-selected="true"] { color: #E9B872; }
[data-baseweb="tab-highlight"] { background: #E9B872 !important; }
@media (max-width: 900px) {
  .readout { grid-template-columns: repeat(2, 1fr); }
  .ro:nth-child(odd) { border-left: 0; padding-left: 0; }
  .hero { flex-wrap: wrap; } .hero-src { margin-left: 0; text-align: left; } .hero-title { font-size: 2rem; }
}
</style>
"""

_MARK = ("<svg class='mark' viewBox='0 0 64 64' role='img' aria-label='Crescent Moon with Earth'>"
         "<circle cx='32' cy='34' r='26' fill='#E6E2D8'/><circle cx='43' cy='28' r='25' fill='#101318'/>"
         "<circle cx='52' cy='12' r='4.5' fill='#6FB1E8'/></svg>")


def hero_html(eph_src: str, eph_real: bool, dem_src: str, dem_real: bool) -> str:
    dot = lambda ok: f"<i class='dot{'' if ok else ' warn'}'></i>"
    return (f"<div class='hero'>{_MARK}<div><div class='hero-title'>Lunar Landing Site Navigator</div>"
            "<div class='hero-sub'>Sunlight, Earth line-of-sight and landing risk at the Moon's south pole. "
            "Team Schnauzers, NASA Space Apps 2026.</div></div>"
            f"<div class='hero-src'>{dot(eph_real)}Ephemeris: {eph_src}<br>{dot(dem_real)}Terrain: {dem_src}</div></div>")


def readout_html(items) -> str:
    """items: (label, value, unit, note) tuples."""
    cells = "".join(f"<div class='ro'><div class='ro-l'>{l}</div><div class='ro-v'>{v}<span>{u}</span></div>"
                    f"<div class='ro-n'>{n}</div></div>" for l, v, u, n in items)
    return f"<div class='readout'>{cells}</div>"


# ----------------------------------------------------------------------------
# Shared building blocks
# ----------------------------------------------------------------------------
def _axis(title, **kw):
    return dict(title=dict(text=title, font=dict(size=11)), backgroundcolor="rgba(26,30,37,0.55)", showbackground=True,
                gridcolor="#2c323c", zerolinecolor="#2c323c", color=MUTED, showspikes=False, **kw)


def _scene(xt, yt, zt, aspect, eye, **zkw):
    return dict(xaxis=_axis(xt), yaxis=_axis(yt), zaxis=_axis(zt, **zkw), aspectmode="manual",
                aspectratio=dict(x=aspect[0], y=aspect[1], z=aspect[2]), bgcolor="rgba(0,0,0,0)",
                camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]), up=dict(x=0, y=0, z=1)))


def _base(fig, height, scene):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", scene=scene,
                      font=dict(family=FONT, color=BONE, size=12),
                      legend=dict(orientation="h", x=0, y=0.02, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor=REGOLITH, bordercolor=DUST, font=dict(family=FONT, color=BONE)))
    return fig


def _plane(x0, x1, y0, y1, z, opacity=0.14):
    """Translucent horizontal plane: the level surface z = const."""
    return go.Surface(x=[x0, x1], y=[y0, y1], z=[[z, z], [z, z]], opacity=opacity, showscale=False,
                      colorscale=[[0, "#ffffff"], [1, "#ffffff"]], hoverinfo="skip", showlegend=False, lighting=FLAT)


def _level(z, width=6):
    """Contour drawn only at height z (on the surface) and projected onto the floor."""
    return dict(z=dict(show=True, start=float(z), end=float(z), size=1, color="#ffffff", width=width,
                       usecolormap=False, project=dict(z=True)))


def _cbar(title):
    return dict(title=dict(text=title, side="right"), thickness=9, len=0.6, tickfont=dict(color=MUTED))


def _diverging(z0, kind):
    """Hard colour break at the 0° level: blues below (shadow / no link), gold or earth-blue above."""
    z0 = float(np.clip(z0, 0.1, 0.9))
    lo = ["#06080d", "#1b2740", "#3d5a85"]
    hi = ["#8a6a2f", "#d6a24f", "#fff0c2"] if kind == "sun" else ["#1f6f8b", "#6fb1e8", "#d8efff"]
    return [[0, lo[0]], [z0 * 0.5, lo[1]], [z0 - 0.002, lo[2]], [z0, hi[0]], [0.5 * (1 + z0), hi[1]], [1, hi[2]]]


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_spacetime(pos, days, margin, kind="sun", xlabel="West → East (m)", lo=-8.0, hi=3.0):
    """Surface: height = elevation of the Sun/Earth above the local horizon, over (position, time).
    The white contour at 0° is the shadow / line-of-sight boundary."""
    m = np.clip(margin, lo, hi)
    surf = go.Surface(x=pos, y=days, z=m, surfacecolor=m, cmin=lo, cmax=hi, colorscale=_diverging(-lo / (hi - lo), kind),
                      lighting=SOFT, colorbar=_cbar("Margin (°)"), contours=_level(0.0),
                      hovertemplate="%{x:.0f} m, day %{y:.1f}<br>margin %{z:.2f}°<extra></extra>")
    c = len(pos) // 2
    target = go.Scatter3d(x=np.full(len(days), pos[c]), y=days, z=m[:, c] + 0.08, mode="lines",
                          line=dict(color=BONE, width=4), name="Nominal target", hoverinfo="skip")
    fig = go.Figure([surf, _plane(pos[0], pos[-1], days[0], days[-1], 0.0), target])
    return _base(fig, 560, _scene(xlabel, "Mission day", "Elevation margin (°)", (1, 1.15, 0.5), (1.7, -1.6, 0.9), range=[lo, hi]))


def fig_terrain(gx, gy, z, shade, vexag=2.0):
    """Terrain relief lit by the Sun at the cursor epoch (with cast shadows) and level curves projected below."""
    relief = float(z.max() - z.min())
    step = max(5.0, round(relief / 12 / 5) * 5.0)
    surf = go.Surface(x=gx, y=gy, z=z, surfacecolor=shade, colorscale=LUNAR, cmin=0, cmax=1, showscale=False, lighting=FLAT,
                      contours=dict(z=dict(show=True, start=float(np.floor(z.min() / step) * step), end=float(z.max()), size=step,
                                           color="rgba(230,226,216,0.22)", width=1, usecolormap=False, project=dict(z=True))),
                      hovertemplate="E %{x:.0f} m, N %{y:.0f} m<br>elevation %{z:.1f} m<extra></extra>")
    c = z.shape[0] // 2
    target = go.Scatter3d(x=[0], y=[0], z=[z[c, c] + 8], mode="markers", marker=dict(size=7, color=BONE, symbol="diamond"),
                          name="Target", hoverinfo="skip")
    zr = max(relief * vexag / (gx[-1] - gx[0]), 0.08)
    return _base(go.Figure([surf, target]), 520, _scene("East (m)", "North (m)", "Elevation (m)", (1, 1, zr), (1.5, -1.6, 0.9)))


def fig_sky(az, hor, sun, earth, idx, floor=-2.0):
    """Cylindrical sky panorama: terrain horizon as a wall, Sun and Earth tracks above it.
    sun / earth: dicts with az, el, vis arrays. idx: epoch cursor."""
    r = lambda a, rad=1.0: (rad * np.sin(np.radians(a)), rad * np.cos(np.radians(a)))
    X, Y = r(az)
    top = max(8.0, float(np.nanmax(sun["el"])) + 1, float(np.nanmax(earth["el"])) + 1)
    wall = go.Surface(x=np.vstack([X, X]), y=np.vstack([Y, Y]), z=np.vstack([np.full_like(hor, floor), hor]),
                      surfacecolor=np.vstack([hor, hor]), colorscale=LUNAR, cmin=0, cmax=max(float(hor.max()), 1.0),
                      showscale=False, opacity=0.78, lighting=FLAT, hoverinfo="skip")
    skyline = go.Scatter3d(x=X, y=Y, z=hor, mode="lines", line=dict(color=BONE, width=4), name="Terrain horizon", hoverinfo="text",
                           text=[f"azimuth {a:.0f}°, horizon {h:.2f}°" for a, h in zip(az, hor)])
    t = np.linspace(0, 2 * np.pi, 181)
    ring = go.Scatter3d(x=np.sin(t), y=np.cos(t), z=np.zeros_like(t), mode="lines", line=dict(color=MUTED, width=2, dash="dash"),
                        name="Geometric horizon (0°)", hoverinfo="skip")
    compass = go.Scatter3d(x=[0, 1.2, 0, -1.2], y=[1.2, 0, -1.2, 0], z=[0] * 4, mode="text", text=["N", "E", "S", "W"],
                           textfont=dict(color=MUTED, size=13), hoverinfo="skip", showlegend=False)
    traces = [wall, skyline, ring, compass]
    for body, color, dim, label in ((sun, SUN, "#6e5a3a", "Sun"), (earth, EARTH, "#2f4a63", "Earth")):
        x, y = r(body["az"], 1.03)
        traces.append(go.Scatter3d(x=x, y=y, z=body["el"], mode="markers", marker=dict(size=2.6, color=np.where(body["vis"], color, dim)),
                                   name=f"{label} track (dim = hidden)", hoverinfo="skip"))
        traces.append(go.Scatter3d(x=[x[idx]], y=[y[idx]], z=[body["el"][idx]], mode="markers", name=f"{label} now", hoverinfo="skip",
                                   marker=dict(size=9, color=color, symbol="diamond", line=dict(color="#ffffff", width=2))))
    scene = _scene("", "", "Elevation (°)", (1, 1, 0.5), (1.2, -2.0, 0.75), range=[floor, top])
    scene["xaxis"], scene["yaxis"] = dict(visible=False), dict(visible=False)
    return _base(go.Figure(traces), 520, scene)


def fig_uptime(gx, gy, up, mc, target):
    """Surface: height = solar uptime at each landing point. The plane is the requirement; the white
    contour is where the surface crosses it. Dispersion samples and σ-ellipses are draped on top."""
    lift = 1.5
    surf = go.Surface(x=gx, y=gy, z=up, surfacecolor=up, cmin=0, cmax=100, colorscale=UPTIME, lighting=SOFT,
                      colorbar=_cbar("Solar uptime (%)"), contours=_level(target),
                      hovertemplate="E %{x:.0f} m, N %{y:.0f} m<br>uptime %{z:.1f}%<extra></extra>")
    traces = [surf, _plane(gx[0], gx[-1], gy[0], gy[-1], target)]
    s, ok, u = mc["samples"], mc["ok"], mc["uptime"]
    for mask, color, name in ((ok, SAFE, "Meets requirement"), (~ok, RISK, "Below requirement")):
        traces.append(go.Scatter3d(x=s[mask, 0], y=s[mask, 1], z=u[mask] + lift, mode="markers", name=name, hoverinfo="skip",
                                   marker=dict(size=2.4, color=color, opacity=0.85)))
    for k, color in ((1, BONE), (2, EARTH), (3, SUN)):
        ex, ey = ill.ellipse_xy(mc["cov"], k)
        traces.append(go.Scatter3d(x=ex, y=ey, z=ill.sample_grid(gx, gy, up, ex, ey) + lift, mode="lines",
                                   line=dict(color=color, width=4), name=f"{k}σ ellipse", hoverinfo="skip"))
    u0 = float(ill.sample_grid(gx, gy, up, [0.0], [0.0])[0])
    traces.append(go.Scatter3d(x=[0], y=[0], z=[u0 + lift], mode="markers", name="Nominal target", hoverinfo="skip",
                               marker=dict(size=6, color="#ffffff", symbol="cross")))
    return _base(go.Figure(traces), 560, _scene("Cross-track ΔX (m)", "Down-track ΔY (m)", "Solar uptime (%)", (1, 1, 0.55), (1.5, -1.7, 0.95), range=[0, 100]))


def fig_response(sx, sy, P, now, level=90.0):
    """Landing-precision response surface: P(success) for every (σx, σy). The white contour marks the
    precision needed for `level` % confidence; the pin is the current dispersion."""
    surf = go.Surface(x=sx, y=sy, z=P, surfacecolor=P, cmin=0, cmax=100, colorscale=RESPONSE, lighting=SOFT,
                      colorbar=_cbar("P(success) %"), contours=_level(level),
                      hovertemplate="σx %{x:.0f} m, σy %{y:.0f} m<br>P(success) %{z:.1f}%<extra></extra>")
    x0, y0, p0 = now
    pin = go.Scatter3d(x=[x0, x0], y=[y0, y0], z=[0, p0], mode="lines+markers", name="Current dispersion", hoverinfo="skip",
                       line=dict(color=BONE, width=4), marker=dict(size=[1, 7], color=BONE))
    fig = go.Figure([surf, _plane(sx[0], sx[-1], sy[0], sy[-1], level), pin])
    return _base(fig, 560, _scene("σx cross-track (m)", "σy down-track (m)", "P(success) %", (1, 1, 0.55), (1.6, -1.6, 0.9), range=[0, 100]))

