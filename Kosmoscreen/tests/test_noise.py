from dataclasses import replace
import json

import numpy as np
import pytest
import rasterio

from landscape.demo import create_demo
from landscape.noise import (GradientNoise2D, PERLIN_DOI, PBRT_NOISE_URL,
                             normalized_octave_noise, permutation_for_seed, terrain_heights)
from landscape.pipeline import build_model
from landscape.project import Project


def test_noise_repeatability_seed_and_lattice():
    x, y = np.meshgrid(np.linspace(-3.7, 3.1, 43), np.linspace(-2.4, 2.8, 37))
    a = GradientNoise2D(permutation_for_seed(42))
    b = GradientNoise2D(permutation_for_seed(42))
    c = GradientNoise2D(permutation_for_seed(43))
    assert np.array_equal(a(x, y), b(x, y))
    assert not np.allclose(a(x, y), c(x, y))
    assert a(np.arange(-300, 300), 0).tolist() == [0.0] * 600
    assert np.allclose(a(x, y), a(x + 256, y - 256), atol=1e-12)
    assert np.isfinite(terrain_heights(x * 600, y * 600, a)).all()


def test_noise_manually_calculated_midpoint():
    # При тождественной перестановке градиенты по углам ячейки направлены на восток, северо-восток, северо-восток и север.
    # В центре ячейки веса равны 1/4, а проекции равны 1/2, 0, 0 и −1/2.
    noise = GradientNoise2D(list(range(256)))
    assert float(noise(0.5, 0.5)) == pytest.approx(0, abs=1e-15)
    # В точке (1/2, 0) усредняются две проекции: 1/2 и −1/(2√2).
    assert float(noise(0.5, 0)) == pytest.approx((1 - 1 / np.sqrt(2)) / 4)


@pytest.mark.parametrize("axis", [0, 1])
def test_noise_first_and_second_derivatives_match_across_cells(axis):
    noise = GradientNoise2D(permutation_for_seed(42))
    h = 1e-4
    for boundary in (-2, 0, 1):
        def value(offset):
            coords = [0.37, 0.37]
            coords[axis] = boundary + offset
            return float(noise(*coords))
        f = value(0)
        left_first = (f - value(-h)) / h
        right_first = (value(h) - f) / h
        left_second = (f - 2 * value(-h) + value(-2*h)) / h**2
        right_second = (value(2*h) - 2 * value(h) + f) / h**2
        assert left_first == pytest.approx(right_first, abs=1e-5)
        assert left_second == pytest.approx(right_second, abs=0.03)


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="Seed"):
        permutation_for_seed(seed)


def test_perlin_demo_provenance_and_pipeline(tmp_path):
    project = Project.load(create_demo(tmp_path / "perlin", seed=17))
    metadata = project.data["generator"]
    assert metadata["algorithm_reference"] == PERLIN_DOI
    assert [item["url"] for item in metadata["algorithm_references"]] == [PERLIN_DOI, PBRT_NOISE_URL]
    assert {PERLIN_DOI, PBRT_NOISE_URL} <= {item["url"] for item in project.data["sources"]}
    assert metadata["seed"] == 17
    assert metadata["permutation"] == permutation_for_seed(17)
    with rasterio.open(project.file("dem")) as src:
        assert src.crs.to_epsg() == 32638
        assert src.res == (30, 30)
        assert json.loads(src.tags()["generator_metadata"]) == metadata
        dem = src.read(1, masked=True)
        assert dem.mask.sum() == 1
        rows, cols = np.indices(dem.shape)
        # Восстанавливаем высоты по сохранённой перестановке, не генерируя её заново.
        heights = terrain_heights((cols + 0.5) * 30 - 660, 660 - (rows + 0.5) * 30,
                                  GradientNoise2D(metadata["permutation"]))
        assert np.array_equal(dem.compressed(), heights.astype(np.float32)[~dem.mask])
    project.parameters = replace(project.parameters, grid_nodes=31, max_trees=5)
    scene, mesh, report = build_model(project, preview=True)
    assert mesh is None and report["status"] == "scene_ready"
    assert report["object_counts"] == {"buildings_added": 8, "trees_added": 5, "skipped": 0}
    assert scene.elevation["maximum_m"] > scene.elevation["minimum_m"]


def test_invalid_method_does_not_write(tmp_path):
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Метод"):
        create_demo(destination, method="unknown")
    assert not destination.exists()


def test_octave_synthesis_against_hand_calculation():
    # Линейная функция позволяет проверить частоты и амплитуды независимо от градиентного шума.
    # При x = 2 и y = 1 базовое значение равно 4. Вклад трёх октав: 4 + 0,5 × 8 + 0,25 × 16 = 12.
    probe = lambda x, y: x + 2 * y
    assert normalized_octave_noise(2, 1, probe, octaves=3, wavelength_m=1) == pytest.approx(12 / 1.75)
    # При изменении множителя частоты ожидаемая спектральная сумма также должна измениться.
    assert normalized_octave_noise(2, 1, probe, octaves=2, lacunarity=3,
                                    persistence=0.25, wavelength_m=1) == pytest.approx(7 / 1.25)


def test_octave_normalization_and_single_octave():
    x, y = np.meshgrid(np.linspace(-180, 240, 11), np.linspace(-60, 320, 9))
    noise = GradientNoise2D(permutation_for_seed(7))
    assert np.allclose(normalized_octave_noise(x, y, noise, octaves=1),
                       noise(x / 600, y / 600), rtol=0, atol=1e-14)
    constant = lambda x, y: np.ones_like(x) * 0.75
    assert np.allclose(normalized_octave_noise(x, y, constant, octaves=8), 0.75)


@pytest.mark.parametrize("kwargs", [{"octaves": 0}, {"octaves": 2.5}, {"octaves": True},
                                    {"persistence": 0}, {"lacunarity": float("nan")},
                                    {"wavelength_m": -1}])
def test_invalid_spectral_parameters(kwargs):
    with pytest.raises(ValueError):
        normalized_octave_noise(0, 0, lambda x, y: x, **kwargs)
