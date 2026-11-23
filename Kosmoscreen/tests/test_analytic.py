from dataclasses import replace
import json
import math

import numpy as np
import pytest
import rasterio

from landscape.analytic import FRANKE_DOI, analytic_heights, franke
from landscape.demo import create_demo
from landscape.geodata import bilinear
from landscape.mesh import require_valid
from landscape.pipeline import build_model
from landscape.project import Project


def test_franke_known_corner_and_broadcast():
    # Подставляем (0, 0) в опубликованную формулу. Во втором слагаемом показатель степени линеен по y.
    expected = (0.75 * math.exp(-2) + 0.75 * math.exp(-1/49 - 1/10)
                + 0.5 * math.exp(-14.5) - 0.2 * math.exp(-65))
    assert float(franke(0, 0)) == pytest.approx(expected, abs=1e-14)
    assert franke(np.zeros((3, 1)), np.zeros(4)).shape == (3, 4)
    assert np.allclose(franke(np.zeros((3, 1)), np.zeros(4)), expected)
    # В точке (1, 1) показатели экспонент равны −24,5, −100/49 − 1, −10 и −29.
    expected_end = (0.75 * math.exp(-24.5) + 0.75 * math.exp(-100/49 - 1)
                    + 0.5 * math.exp(-10) - 0.2 * math.exp(-29))
    assert float(franke(1, 1)) == pytest.approx(expected_end, abs=1e-14)


def test_mapping_and_halo():
    assert analytic_heights(-500, -500) == pytest.approx(800 + 120 * franke(0, 0))
    assert analytic_heights(500, 500) == pytest.approx(800 + 120 * franke(1, 1))
    assert analytic_heights(0, 0) == pytest.approx(800 + 120 * franke(0.5, 0.5))
    assert analytic_heights(645, 645) != analytic_heights(500, 500)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_coordinates(value):
    with pytest.raises(ValueError, match="конечными"):
        franke(value, 0)


def test_interpolation_converges_to_analytic_reference():
    axis = np.linspace(-498.75, 498.75, 400)
    x, y = np.meshgrid(axis, axis)
    exact = analytic_heights(x, y)
    errors = []
    for step, size in ((30, 44), (15, 88)):
        rows, cols = np.indices((size, size))
        dem = analytic_heights((cols + 0.5)*step - 660, 660 - (rows + 0.5)*step)
        sampled = bilinear(dem, (x + 660)/step - 0.5, (660 - y)/step - 0.5)
        errors.append(np.sqrt(np.mean((sampled - exact)**2)))
    assert errors[0] < 1.0
    assert errors[1] < errors[0] / 3  # При уменьшении шага ожидаем сходимость примерно второго порядка.


def test_franke_dataset_full_pipeline(tmp_path):
    project = Project.load(create_demo(tmp_path / "franke", method="analytic"))
    with rasterio.open(project.file("dem")) as src:
        data = src.read(1, masked=True)
        metadata = json.loads(src.tags()["generator_metadata"])
        assert src.crs.to_epsg() == 32638 and src.res == (30, 30)
        assert data.mask.sum() == 1 and data.mask[17, 18]
        assert metadata == project.data["generator"]
        assert metadata["algorithm_references"][0]["url"] == FRANKE_DOI
        rows, cols = np.indices(data.shape)
        expected = analytic_heights((cols+0.5)*30-660, 660-(rows+0.5)*30)
        assert np.array_equal(data.compressed(), expected.astype(np.float32)[~data.mask])
    project.parameters = replace(project.parameters, grid_nodes=31, max_trees=5)
    scene, mesh, report = build_model(project)
    assert report["dem"]["missing_inside"] == 1
    assert report["object_counts"] == {"buildings_added": 8, "trees_added": 5, "skipped": 0}
    assert require_valid(mesh)["components"] == 1
    assert report["validation"]["dimensions_mm"][:2] == pytest.approx([180, 180])


def test_analytic_does_not_depend_on_seed(tmp_path):
    paths = [create_demo(tmp_path / str(seed), method="analytic", seed=seed) for seed in (1, 73)]
    with rasterio.open(paths[0].parent / "dem.tif") as first, rasterio.open(paths[1].parent / "dem.tif") as second:
        assert np.array_equal(first.read(), second.read())


@pytest.mark.parametrize("method,directory", [("analytic", "franke"), ("perlin", "perlin"),
                                              ("analytic-legacy", "analytic-legacy")])
def test_cli_default_directory_depends_on_method(tmp_path, monkeypatch, method, directory):
    from landscape.__main__ import main
    monkeypatch.chdir(tmp_path)
    assert main(["demo", "--method", method]) == 0
    assert (tmp_path / "examples" / directory / "project.json").exists()
