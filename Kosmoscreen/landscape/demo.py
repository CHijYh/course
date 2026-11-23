"""Воспроизводимые искусственные наборы данных для проверки построения рельефа."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform

from .project import Parameters
from .analytic import FRANKE_DOI, FRANKE_CATALOG, analytic_heights, analytic_metadata
from .noise import GradientNoise2D, PERLIN_DOI, PBRT_NOISE_URL, generator_metadata, permutation_for_seed, terrain_heights


def create_demo(destination, *, method="perlin", seed=42):
    """Создать тестовые данные по шуму Перлина, функции Франке или прежней модели плоскости с холмом."""
    if method not in ("perlin", "analytic", "analytic-legacy"):
        raise ValueError("Метод генерации должен быть perlin, analytic или analytic-legacy.")
    permutation = permutation_for_seed(seed)
    noise = GradientNoise2D(permutation)
    metadata = (generator_metadata(seed, permutation) if method == "perlin" else
                {"method": "analytic", "implementation_version": 1,
                 "description": "Inclined plane plus Gaussian hill; original project fixture."})
    if method == "analytic":
        metadata = analytic_metadata()
    folder = Path(destination).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    cx, cy = transform("EPSG:4326", "EPSG:32638", [42.730], [43.907])
    cx, cy = cx[0], cy[0]
    def write_raster(name, step, size, rgb=False):
        affine = from_origin(cx-size*step/2, cy+size*step/2, step, step)
        rows, cols = np.indices((size, size))
        x, y = (cols+0.5)*step-size*step/2, size*step/2-(rows+0.5)*step
        if rgb:
            road = np.abs(y+0.23*x+40) < 12
            green = 0.5+0.5*np.sin(x/90)*np.cos(y/120)
            values = np.stack((70+45*green, 100+50*green, 55+30*green)).astype(np.uint8)
            values[:, road] = np.array([175, 168, 149], dtype=np.uint8)[:, None]
        else:
            if method == "perlin":
                heights = terrain_heights(x, y, noise)
            elif method == "analytic":
                heights = analytic_heights(x, y)
            else:
                heights = 850 + 0.09*x + 0.06*y + 55*np.exp(-((x-120)**2+(y-80)**2)/85000)
            values = heights.astype(np.float32)[None, :, :]
            values[0, 17, 18] = -9999  # Пропуск внесён намеренно. Его доля ниже допустимых 5 %.
        with rasterio.open(folder/name, "w", driver="GTiff", width=size, height=size,
                           count=values.shape[0], dtype=values.dtype, crs="EPSG:32638", transform=affine,
                           nodata=None if rgb else -9999, compress="deflate") as out:
            out.write(values)
            out.update_tags(source="SYNTHETIC; NOT Sentinel-2 or FABDEM",
                            generator=("project decorative texture" if rgb else metadata["method"]))
            if not rgb:
                out.update_tags(generator_metadata=json.dumps(metadata))
    write_raster("dem.tif", 30, 44)
    write_raster("rgb.tif", 10, 132, rgb=True)
    def geographic(points):
        x = [cx-500+p[0] for p in points]
        y = [cy-500+p[1] for p in points]
        lon, lat = transform("EPSG:32638", "EPSG:4326", x, y)
        return [list(p) for p in zip(lon, lat)]
    features = []
    for i, (x, y) in enumerate(((140, 190), (245, 225), (355, 210), (480, 260), (580, 230), (680, 200), (280, 410), (420, 430))):
        poly = [(x, y), (x+45, y), (x+45, y+35), (x, y+35), (x, y)]
        if i == 7:
            poly = [(x,y), (x+65,y), (x+65,y+25), (x+30,y+25), (x+30,y+60), (x,y+60), (x,y)]
        props = {"building": "yes", "source": "synthetic footprint"}
        if i % 3 == 0:
            props["height"] = str(9+i)+" m"
        elif i % 3 == 1:
            props["building:levels"] = "3"
        features.append({"type": "Feature", "id": f"demo_building_{i+1}", "properties": props,
                         "geometry": {"type": "Polygon", "coordinates": [geographic(poly)]}})
    for i, point in enumerate(((110, 320), (550, 420), (760, 250), (110, 200))):
        features.append({"type": "Feature", "id": f"demo_tree_{i+1}",
                         "properties": {"natural": "tree", "source": "synthetic manual point"},
                         "geometry": {"type": "Point", "coordinates": geographic([point])[0]}})
    forest = [(130, 550), (880, 550), (880, 900), (130, 900), (130, 550)]
    features.append({"type": "Feature", "id": "demo_forest", "properties": {"landuse": "forest", "source": "synthetic area"},
                     "geometry": {"type": "Polygon", "coordinates": [geographic(forest)]}})
    (folder/"objects.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")
    project = {"version": 1, "title": "Kosmoscreen — искусственный тестовый участок",
               "dataset_kind": "synthetic; not geographic evidence",
               "generator": metadata,
               "area": {"center": [42.730, 43.907], "width_m": 1000, "height_m": 1000, "crs": "EPSG:32638"},
               "inputs": {"dem": "dem.tif", "rgb": "rgb.tif", "objects": "objects.geojson"},
               "output": "output", "parameters": asdict(Parameters()),
               "sources": [{"name": "Analytical synthetic fixture", "url": "landscape/demo.py", "license": "CC0-1.0",
                            "description": "Artificial elevations, colours and objects. Coordinates locate a test frame only."}]}
    if method == "analytic":
        project["title"] = "Kosmoscreen — аналитическая поверхность Франке"
        project["sources"] = [
            {"name": "Franke R. Scattered Data Interpolation: Tests of Some Methods (1982)",
             "url": FRANKE_DOI, "description": "Analytical benchmark function; coordinate and height scaling are project-specific."},
            {"name": "Surjanovic S., Bingham D. Virtual Library: Franke's Function",
             "url": FRANKE_CATALOG, "description": "Function definition and unit-square domain; independent NumPy evaluation of the formula."},
            {"name": "Kosmoscreen synthetic data", "url": "landscape/demo.py", "license": "CC0-1.0",
             "description": "Synthetic DEM, decorative RGB, artificial buildings and vegetation; no geographic evidence."}]
    elif method == "perlin":
        project["title"] = "Kosmoscreen — синтетический рельеф по методу Перлина"
        project["sources"] = [
            {"name": "Perlin K. Improving Noise (2002)", "url": PERLIN_DOI,
             "description": "Algorithm reference for gradient noise and quintic fade; independent 2D adaptation, not the author's reference code."},
            {"name": "Pharr M., Jakob W., Humphreys G. Physically Based Rendering, 3rd ed., section 10.6.3",
             "url": PBRT_NOISE_URL,
             "description": "Spectral synthesis reference: octave frequency scaling and amplitude weighting. Independent normalized 2D implementation, without PBRT's derivative-based filtering."},
            {"name": "Kosmoscreen synthetic data", "url": "landscape/demo.py", "license": "CC0-1.0",
             "description": "Generated DEM; project-specific octave settings, decorative RGB and artificial objects. No satellite measurements."}]
    (folder/"project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder/"project.json"

