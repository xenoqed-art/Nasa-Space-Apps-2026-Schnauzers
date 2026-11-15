# 🌑 Lunar Landing Site Navigator
**South Pole Illumination, Direct-to-Earth Line-of-Sight & Probabilistic Risk Analyzer**  
*Built by Team Schnauzers — 2026 NASA International Space Apps Challenge*

---

## Overview
Missions targeting the Moon's south pole operate under extreme illumination and communication constraints. Because the Moon's axial tilt is only ~1.54°, sunlight grazes the horizon nearly horizontally, casting long, permanent shadows across crater floors while illuminating elevated crater rims for extended periods. Meanwhile, Earth experiences slow apparent vertical oscillations (lunar libration) near the horizon, periodically interrupting Direct-to-Earth (DTE) radio links.

**Lunar Landing Site Navigator** transforms raw orbital ephemerides and lunar elevation maps into an intuitive, interactive decision-support dashboard. It enables mission planners, researchers, and students to compare candidate landing sites and mission dates in seconds—integrating **deterministic orbital physics**, **local topographic horizon masking**, and **Monte Carlo landing dispersion risk analysis**.

---

## Core Features
* **Celestial Ephemeris Engine:** Computes exact topocentric azimuth and elevation angles for both the Sun and Earth from any lunar south pole coordinate $(\text{lat}, \text{lon})$ across custom mission timelines.
* **Topographic Horizon Occlusion & Power Estimation:** Models 360° local horizon elevation profiles from lunar Digital Elevation Models (DEM) to determine real solar and Earth visibility, converting solar incidence angles into estimated photovoltaic power ($W/m^2$).
* **Monte Carlo Landing Dispersion Analysis:** Simulates spatial landing uncertainty using a bivariate normal dispersion ellipse $(X, Y) \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ around nominal target coordinates to quantify:
  * Probability of meeting minimum solar power uptime.
  * Probability of maintaining continuous Direct-to-Earth (DTE) communications.
  * Worst-case continuous blackout duration (thermal and battery survival metric).
* **Side-by-Side Site Comparison:** Compares candidate south pole targets (such as Shackleton Crater Rim, Malapert Massif, and Cabeus) across different launch windows using a unified **Mission Viability Score (0–100)**.

---

## Repository Architecture
The codebase is decoupled into four independent modules:

```text
Nasa-Space-Apps-2026-Schnauzers/
├── .streamlit/
│   └── config.toml        # Streamlit UI theme and server configuration
├── src/
│   ├── __init__.py
│   ├── models.py          # Shared data structures (LandingSite, MissionWindow, RiskReport)
│   ├── ephemeris.py       # Solar & Earth topocentric azimuth/elevation calculations
│   ├── terrain.py         # Local horizon profile extraction & photovoltaic power model
│   ├── illumination.py    # Lunar solar illumination & shadow casting calculations
│   ├── visuals.py         # 3D lunar surface rendering & Plotly visualization components
│   └── risk_engine.py     # Monte Carlo landing ellipse simulation & viability scoring
├── data/                  # Candidate coordinates and processed LRO LOLA elevation subsets
├── app.py                 # Interactive Streamlit dashboard & Plotly visualizations
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
