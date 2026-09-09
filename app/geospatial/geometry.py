from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shapely_transform

try:
    import geopandas as gpd
    import pyproj
    import rasterio
    import rasterio.features
    from rasterio.transform import Affine
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False


class GeospatialProcessor:
    """
    Handles conversion of binary segmentation masks into vector polygons,
    computes area in km^2, perimeter in km, geographic centroids, and exports GeoJSON.
    Handles un-georeferenced images gracefully without inventing fake coordinates.
    """

    def __init__(self, default_crs: str = "EPSG:4326"):
        self.default_crs = default_crs

    def process_region_geometry(
        self,
        cleaned_mask: np.ndarray,
        region_info: dict[str, Any],
        meta: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculates geographic or pixel-space geometry for a single detected spill region.

        Returns dict matching GeometryMetadata + MeasurementsMetadata.
        """
        is_georeferenced = meta.get(
            "is_georeferenced", False) and GEOSPATIAL_AVAILABLE
        crs_str = meta.get("crs", None)
        affine_tf = meta.get("transform", None)

        if is_georeferenced and affine_tf is not None:
            # Extract vector shapes using rasterio
            region_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
            coords = region_info["region_mask_coords"]
            region_mask[coords[:, 0], coords[:, 1]] = 1

            # Convert raster mask to shapely geometry in CRS
            shapes = list(rasterio.features.shapes(
                region_mask, mask=(region_mask == 1), transform=affine_tf))
            if not shapes:
                return self._fallback_pixel_geometry(region_info)

            # Combine shapes into a single Polygon or MultiPolygon
            geoms = [shape(geom_dict) for geom_dict, val in shapes if val == 1]
            if not geoms:
                return self._fallback_pixel_geometry(region_info)

            main_poly = geoms[0] if len(geoms) == 1 else MultiPolygon(geoms)

            # Compute Geographic Centroid
            centroid_pt = main_poly.centroid
            lon, lat = centroid_pt.x, centroid_pt.y

            # Compute Area in km^2 and Perimeter in km
            area_km2, perimeter_km = self._calculate_geo_measurements(
                main_poly, crs_str)

            # Bounding box in CRS: [minx, miny, maxx, maxy]
            bounds = list(main_poly.bounds)

            return {
                "is_geographic": True,
                "crs": crs_str or self.default_crs,
                "centroid": {
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "pixel_x": region_info["centroid_pixel"]["x"],
                    "pixel_y": region_info["centroid_pixel"]["y"]
                },
                "bbox": [round(b, 6) for b in bounds],
                "polygon_geojson": mapping(main_poly),
                "area_km2": round(area_km2, 4) if area_km2 is not None else None,
                "perimeter_km": round(perimeter_km, 4) if perimeter_km is not None else None,
                "area_pixels": region_info["area_pixels"],
                "width_pixels": region_info["width_pixels"],
                "height_pixels": region_info["height_pixels"],
                "aspect_ratio": region_info["aspect_ratio"]
            }
        else:
            return self._fallback_pixel_geometry(region_info)

    def _fallback_pixel_geometry(self, region_info: dict[str, Any]) -> dict[str, Any]:
        """
        Creates pixel-space geometry record when raster lacks CRS/transform.
        """
        min_col, min_row, max_col, max_row = region_info["bbox_pixel"]
        pixel_poly = Polygon([
            (min_col, min_row),
            (max_col, min_row),
            (max_col, max_row),
            (min_col, max_row),
            (min_col, min_row)
        ])

        return {
            "is_geographic": False,
            "crs": None,
            "centroid": {
                "latitude": None,
                "longitude": None,
                "pixel_x": region_info["centroid_pixel"]["x"],
                "pixel_y": region_info["centroid_pixel"]["y"]
            },
            "bbox": [float(min_col), float(min_row), float(max_col), float(max_row)],
            "polygon_geojson": mapping(pixel_poly),
            "area_km2": None,
            "perimeter_km": None,
            "area_pixels": region_info["area_pixels"],
            "width_pixels": region_info["width_pixels"],
            "height_pixels": region_info["height_pixels"],
            "aspect_ratio": region_info["aspect_ratio"]
        }

    def _calculate_geo_measurements(self, poly: Polygon, crs_str: str | None) -> tuple[float | None, float | None]:
        """
        Calculates area in km^2 and perimeter in km by projecting geometry to an equal-area CRS (e.g. World Equidistant Cylindrical or UTM).
        """
        try:
            if not crs_str:
                crs_str = "EPSG:4326"

            src_crs = pyproj.CRS(crs_str)
            # Reproject to equal area projection (e.g. World Mollweide / ESRI:54009 or World Equidistant Cylindrical ESRI:54002)
            tgt_crs = pyproj.CRS("ESRI:54009")  # World Mollweide equal area
            project = pyproj.Transformer.from_crs(
                src_crs, tgt_crs, always_xy=True).transform

            poly_projected = shapely_transform(project, poly)
            area_m2 = poly_projected.area
            perimeter_m = poly_projected.length

            area_km2 = area_m2 / 1e6  # 1 km^2 = 1,000,000 m^2
            perimeter_km = perimeter_m / 1e3  # 1 km = 1,000 m

            return float(area_km2), float(perimeter_km)
        except Exception:
            # Fallback estimation if projection transformation fails
            return None, None

    def export_geojson(self, region_records: list[dict[str, Any]], crs_str: str | None = None) -> dict[str, Any]:
        """
        Exports list of region records into a GeoJSON FeatureCollection structure.
        """
        features = []
        for rec in region_records:
            poly_geojson = rec.get("geometry", {}).get("polygon_geojson", None)
            if poly_geojson is None:
                continue

            feature = {
                "type": "Feature",
                "properties": {
                    "spill_id": rec.get("spill_id"),
                    "confidence": rec.get("detection", {}).get("confidence"),
                    "area_km2": rec.get("measurements", {}).get("area_km2"),
                    "perimeter_km": rec.get("measurements", {}).get("perimeter_km"),
                    "area_pixels": rec.get("measurements", {}).get("area_pixels"),
                    "is_geographic": rec.get("geometry", {}).get("is_geographic"),
                    "centroid_lat": rec.get("geometry", {}).get("centroid", {}).get("latitude"),
                    "centroid_lon": rec.get("geometry", {}).get("centroid", {}).get("longitude")
                },
                "geometry": poly_geojson
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": crs_str or self.default_crs
                }
            } if crs_str else None,
            "features": features
        }
