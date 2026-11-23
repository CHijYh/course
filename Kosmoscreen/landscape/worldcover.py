"""Выборка классов земного покрова ESA WorldCover 2021 v200 на сетке проекта в проекции UTM."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from .project import ProjectError


COLLECTION = "ESA WorldCover 2021 v200"
ROOT = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
CITATION = "Zanaga et al. (2022), ESA WorldCover 10 m 2021 v200, CC BY 4.0, doi:10.5281/zenodo.7254221"


def tile_names(crs, bounds):
    west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds)
    spans = [(west, east)] if west <= east else [(west, 180), (-180, east)]
    names = []
    for lat in range(3 * math.floor(south / 3), 3 * math.ceil(north / 3), 3):
        for left, right in spans:
            for lon in range(3 * math.floor(left / 3), 3 * math.ceil(right / 3), 3):
                names.append(f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}")
    return names


def sample_classes(path, grid):
    with rasterio.open(path) as src:
        if src.count != 1 or src.crs is None or not np.issubdtype(src.dtypes[0], np.integer):
            raise ProjectError("WorldCover: нужен одноканальный целочисленный GeoTIFF с системой координат.")
        with WarpedVRT(src, **grid, nodata=0, resampling=Resampling.nearest) as vrt:
            return vrt.read(1, masked=True).filled(0)


def load_worldcover(project, crs, bounds):
    if not (project.parameters.worldcover and project.parameters.trees and project.parameters.max_trees):
        return None, None
    width = math.ceil((bounds[2] - bounds[0]) / 10)
    height = math.ceil((bounds[3] - bounds[1]) / 10)
    grid = dict(crs=crs, width=width, height=height, transform=from_bounds(*bounds, width, height))
    local = project.file("worldcover")
    metadata = {"source": COLLECTION, "citation": CITATION, "class": 10}
    try:
        if local:
            values = sample_classes(local, grid)
            metadata["file"] = project.data["inputs"]["worldcover"]
        else:
            tiles = tile_names(crs, bounds)
            metadata["tiles"] = tiles
            key = hashlib.sha256(json.dumps([COLLECTION, str(crs), bounds]).encode()).hexdigest()[:24]
            cache = project.output / "worldcover" / (key + ".tif")
            if cache.is_file():
                values = sample_classes(cache, grid)
            else:
                values = np.zeros((height, width), np.uint8)
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="25"):
                    for tile in tiles:
                        url = f"{ROOT}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
                        block = sample_classes("/vsicurl/" + url, grid)
                        np.copyto(values, block, where=block != 0)
                if np.any(values):
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    with NamedTemporaryFile(dir=cache.parent, suffix=".tif", delete=False) as tmp:
                        pending = Path(tmp.name)
                    try:
                        with rasterio.open(pending, "w", driver="GTiff", count=1, dtype="uint8",
                                           compress="deflate", nodata=0, **grid) as dst:
                            dst.write(values, 1)
                        os.replace(pending, cache)
                    finally:
                        pending.unlink(missing_ok=True)
    except (OSError, rasterio.errors.RasterioError) as exc:
        raise ProjectError(f"Не удалось прочитать WorldCover для участка. Проверьте GeoTIFF или доступ к ESA WorldCover. {exc}") from exc
    if not np.any(values):
        raise ProjectError("WorldCover: участок не содержит пикселей с данными. Проверьте покрытие карты.")
    metadata["covered_cells"] = int(np.count_nonzero(values))
    metadata["tree_cover_cells"] = int(np.count_nonzero(values == 10))
    return values == 10, metadata


def tree_candidates(mask, excluded_polygons, surface, seed):
    """Выдать по одной случайно смещённой точке на каждую подходящую ячейку в миллиметрах макета."""
    height, width = mask.shape
    xmax, ymax = surface.xs[-1], surface.ys[-1]
    affine = from_bounds(0, 0, xmax, ymax, width, height)
    if excluded_polygons:
        polygons = ({"type": "Polygon", "coordinates": [[*p, p[0]]]} for p in excluded_polygons)
        excluded = rasterize(polygons, out_shape=mask.shape, transform=affine, all_touched=True)
        mask = mask & (excluded == 0)
    indices = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    for index in indices:
        row, col = divmod(int(index), width)
        dx, dy = rng.uniform(0.08, 0.92, size=2)
        yield int(index), ((col + dx) * affine.a, ymax + (row + dy) * affine.e)

