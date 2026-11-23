from dataclasses import replace
import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from landscape.demo import create_demo
from landscape.pipeline import build_model, validate_data
from landscape.project import Project, ProjectError
from landscape.worldcover import load_worldcover, tile_names


def test_tiles_at_seams_and_in_southern_hemisphere():
    assert tile_names("EPSG:4326", (-0.1, -3.1, 0.1, -2.9)) == [
        "S06W003", "S06E000", "S03W003", "S03E000"]
    assert tile_names("EPSG:4326", (3, 0, 6, 3)) == ["N00E003"]


def test_worldcover_pipeline_without_vectors(tmp_path):
    project = Project.load(create_demo(tmp_path / "project"))
    project.parameters = replace(project.parameters, worldcover=True, trees=True, max_trees=12, grid_nodes=31)
    project.data["inputs"].pop("objects")
    crs, bounds = project.coordinates()
    path = project.path.parent / "cover.tif"
    project.data["inputs"]["worldcover"] = path.name
    affine = from_bounds(*bounds, 100, 100)
    values = np.full((100, 100), 30, np.uint8)
    values[:, :50] = 10
    with rasterio.open(path, "w", driver="GTiff", count=1, dtype="uint8", width=100, height=100,
                       crs=crs, transform=affine, nodata=0) as dataset:
        dataset.write(values, 1)
    checked = validate_data(project)
    _, _, report = build_model(project, preview=True)
    assert checked["objects"] == report["objects"]
    assert report["worldcover"]["trees_added"] == 12
    assert report["worldcover"]["tree_cover_cells"] == 5000
    assert "worldcover" in report["input_hashes"]
    points = [item["position_mm"] for item in report["objects"]]
    assert all(0 < x < project.parameters.long_side_mm / 2 for x, _ in points)
    assert all(math.dist(p, q) >= project.parameters.tree_spacing_m * report["scale_mm_per_m"]
               for i, p in enumerate(points) for q in points[:i])
    with rasterio.open(path, "r+") as dataset:
        dataset.write(np.full_like(values, 30), 1)
    assert build_model(project, preview=True)[2]["worldcover"]["trees_added"] == 0
    with rasterio.open(path, "r+") as dataset:
        dataset.write(np.zeros_like(values), 1)
    with pytest.raises(ProjectError, match="покрытие"):
        load_worldcover(project, crs, bounds)


def test_cached_mosaic_and_area_changes(tmp_path, monkeypatch):
    from landscape import worldcover

    project = Project.load(create_demo(tmp_path / "project"))
    project.parameters = replace(project.parameters, worldcover=True)
    crs, bounds = project.coordinates()
    calls = []
    read = worldcover.sample_classes

    def sample(path, grid):
        if not str(path).startswith("/vsicurl/"):
            return read(path, grid)
        calls.append(path)
        block = np.zeros((grid["height"], grid["width"]), np.uint8)
        part = slice(None, 50) if "N00E000" in path else slice(50, None)
        block[:, part] = 10
        return block

    monkeypatch.setattr(worldcover, "tile_names", lambda *args: ["N00E000", "N00E003"])
    monkeypatch.setattr(worldcover, "sample_classes", sample)
    for _ in range(2):
        mask, _ = load_worldcover(project, crs, bounds)
        assert mask.all()
    assert len(calls) == 2
    west, south, east, north = bounds
    load_worldcover(project, crs, (west+10, south, east+10, north))
    assert len(calls) == 4



def test_worldcover_excludes_occupied_cells():
    from types import SimpleNamespace
    from landscape.worldcover import tree_candidates

    surface = SimpleNamespace(xs=[0, 100], ys=[0, 100])
    occupied = [[(-10, -10), (48, -10), (48, 110), (-10, 110)],
                [(40, 51), (110, 51), (110, 110), (40, 110)]]
    candidates = list(tree_candidates(np.ones((10, 10), bool), occupied, surface, 42))
    assert candidates
    assert all(x >= 50 and y <= 50 for _, (x, y) in candidates)
    assert candidates == list(tree_candidates(np.ones((10, 10), bool), occupied, surface, 42))
