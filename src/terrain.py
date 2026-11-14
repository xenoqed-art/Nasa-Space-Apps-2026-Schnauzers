import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy
from src.models import LandingSite


class TerrainEngine:
    CONSTANTE_SOLAR_LUNAR = 1361.0  # W/m^2 (Irradiancia solar en el exterior lunar)
    EFICIENCIA_PANEL = 0.30         # 30% de eficiencia típica para celdas espaciales

    def __init__(self, dem_path: str):
        """
        Inicializa el motor de terreno cargando el archivo DEM ráster (GeoTIFF de LOLA).
        """
        self.dem_path = dem_path
        with rasterio.open(self.dem_path) as src:
            self.elevation_data = src.read(1)  # Matriz 2D de alturas en metros
            self.transform = src.transform     # Transformación afín (píxeles <-> coordenadas espaciales)
            self.crs = src.crs                 # Sistema de referencia de coordenadas
            self.res_x = src.res[0]            # Resolución espacial en metros por píxel (ej. 5m)
            self.bounds = src.bounds

    def compute_horizon_profile(self, lat: float, lon: float, max_dist_km: float = 50.0, az_step: int = 1) -> dict:
        """
        Calcula el perfil de horizonte topográfico en 360 grados alrededor de un sitio (lat, lon).
        Retorna un diccionario {azimut_grados: angulo_horizonte_grados}.
        """
        horizon_profile = {}
        
        # 1. Convertir lat/lon del sitio a coordenadas de píxel (fila, columna) en el ráster DEM
        # Nota: Asume que el DEM está en proyección estereográfica polar sur compatible.
        row, col = rowcol(self.transform, lon, lat) # Ajustar según el sistema de coordenadas del DEM
        
        # Obtener la altitud base del sitio de aterrizaje (z_0)
        try:
            z_0 = self.elevation_data[row, col]
        except IndexError:
            # Si cae fuera del borde del DEM cargado, usar una altitud por defecto o lanzar error
            z_0 = 0.0

        max_pixels = int((max_dist_km * 1000.0) / self.res_x)

        # 2. Iterar sobre cada dirección de azimut (de 0 a 360 grados)
        for az in range(0, 360, az_step):
            az_rad = np.radians(az)
            max_elevation_angle = -90.0

            # Vectores de dirección en coordenadas métricas locales (Este, Norte)
            sin_az = np.sin(az_rad)
            cos_az = np.cos(az_rad)

            # 3. Trazar rayos a lo largo de la distancia (ray-casting)
            for step in range(1, max_pixels):
                dist_m = step * self.res_x
                # Desplazamiento en píxeles aproximado
                dx = (sin_az * dist_m) / self.res_x
                dy = (cos_az * dist_m) / self.res_x  # Norte positivo

                r_sample = int(row - dy)  # El norte disminuye en filas de matriz ráster
                c_sample = int(col + dx)

                # Verificar límites de la matriz DEM
                if (0 <= r_sample < self.elevation_data.shape[0]) and (0 <= c_sample < self.elevation_data.shape[1]):
                    z_terrain = self.elevation_data[r_sample, c_sample]
                    
                    # Calcular el ángulo de elevación desde el sitio hacia el punto muestreado
                    # angulo = arctan( (z_terreno - z_sitio) / distancia )
                    angle = np.degrees(np.arctan2((z_terrain - z_0), dist_m))

                    if angle > max_elevation_angle:
                        max_elevation_angle = angle
                else:
                    # El rayo salió del mapa DEM disponible
                    break

            horizon_profile[az] = max_elevation_angle

        return horizon_profile

    def evaluate_visibility_and_power(self, site: LandingSite, ephem_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cruza el perfil de horizonte topográfico con la tabla de efemérides temporales
        para determinar visibilidad real (Sol/Tierra) y potencia fotovoltaica generada.
        """
        # 1. Calcular el perfil de horizonte local para el sitio
        horizon = self.compute_horizon_profile(site.lat, site.lon)

        sol_visible_list = []
        tierra_visible_list = []
        potencia_list = []

        # 2. Evaluar cada instante temporal de la misión
        for _, row in ephem_df.iterrows():
            # Redondear azimut al entero más cercano para buscar en el diccionario de horizonte
            az_sol_int = int(round(row["sun_az"])) % 360
            az_tierra_int = int(round(row["earth_az"])) % 360

            h_horizonte_sol = horizon.get(az_sol_int, 0.0)
            h_horizonte_tierra = horizon.get(az_tierra_int, 0.0)

            # Visibilidad geométrica real descontando la topografía
            sol_vis = row["sun_el"] > h_horizonte_sol
            tierra_vis = row["earth_el"] > h_horizonte_tierra

            sol_visible_list.append(sol_vis)
            tierra_visible_list.append(tierra_vis)

            # 3. Cálculo de potencia fotovoltaica (W/m^2)
            if sol_vis and row["sun_el"] > 0:
                # Potencia proporcional al seno del ángulo de elevación solar real
                potencia = self.CONSTANTE_SOLAR_LUNAR * self.EFICIENCIA_PANEL * max(0.0, np.sin(np.radians(row["sun_el"])))
            else:
                potencia = 0.0

            potencia_list.append(potencia)

        # 4. Enriquecer y retornar el DataFrame de resultados
        result_df = ephemeris_df.copy()
        result_df["sol_visible"] = sol_visible_list
        result_df["tierra_visible"] = tierra_visible_list
        result_df["potencia_W"] = potencia_list

        return result_df
