from collections import Counter
import json

import numpy as np
import pytest

from landscape.contours import contour_lines, terrain_heights
from landscape.pipeline import Scene, save_scene, load_scene
from landscape.project import ProjectError
from landscape.terrain import Surface, terrain_mesh


def plane(boost=1):
    heights = np.tile(np.arange(3, dtype=float), (3, 1))
    mesh = terrain_mesh(Surface(np.arange(3), np.arange(3), 3+boost*heights,
                                1, (0, 0, 2, 2), 100, 102))
    return Scene(mesh, [], np.zeros((2, 2, 3), dtype=np.uint8), (2, 2),
                 {"minimum_m": 100, "maximum_m": 102, "scale_mm_per_m": 1,
                  "base_mm": 3, "exaggeration": boost})


def test_plane_contours_shared_edges_and_no_base():
    scene = plane()
    contours = contour_lines(scene.terrain, terrain_heights(scene), 1)
    assert [c.elevation_m for c in contours] == [100, 101, 102]
    for contour in contours:
        assert contour.segments.shape == (2, 2, 3)
        assert np.allclose(contour.segments[:, :, 0], contour.elevation_m-100)
        assert np.allclose(contour.segments[:, :, 2], 3+contour.elevation_m-100)
        assert np.linalg.norm(np.diff(contour.segments, axis=1), axis=2).sum() == pytest.approx(2)


def test_interpolated_contours_and_exaggeration():
    a, b = plane(), plane(3)
    first = contour_lines(a.terrain, terrain_heights(a), .5)
    boosted = contour_lines(b.terrain, terrain_heights(b), .5)
    for x, y in zip(first, boosted):
        assert x.elevation_m == y.elevation_m
        assert np.allclose(x.segments[:, :, :2], y.segments[:, :, :2])
        assert np.allclose(y.segments[:, :, 2], 3+3*(x.elevation_m-100))


def test_hill_is_closed_and_flat_has_no_arbitrary_lines():
    z = np.full((3, 3), 3.0)
    z[1, 1] = 5
    mesh = terrain_mesh(Surface(np.arange(3), np.arange(3), z, 1, (0, 0, 2, 2), 3, 5))
    middle = next(c for c in contour_lines(mesh, mesh.vertices[:, 2], 1) if c.elevation_m == 4)
    degree = Counter(tuple(p) for p in middle.segments.reshape(-1, 3))
    assert set(degree.values()) == {2}
    mesh.vertices[:9, 2] = 3
    assert contour_lines(mesh, mesh.vertices[:, 2], 1) == []


def test_metadata_is_portable_without_sidecar(tmp_path):
    scene = plane(2)
    path = tmp_path / "scene.npz"
    save_scene(scene, None, path)
    loaded, _ = load_scene(path)
    assert loaded.elevation == scene.elevation
    assert np.allclose(terrain_heights(loaded), terrain_heights(scene))


def test_legacy_metadata_and_mismatch(tmp_path):
    scene = plane()
    scene.elevation = None
    path = tmp_path / "scene.npz"
    with pytest.raises(ProjectError, match="Постройте"):
        terrain_heights(scene, path)
    report = {"elevation_range_m": [100, 102], "scale_mm_per_m": 1,
              "parameters": {"base_mm": 3, "exaggeration": 1}}
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    assert terrain_heights(scene, path)[0] == 100
    report["parameters"]["exaggeration"] = 2
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ProjectError):
        terrain_heights(scene, path)


@pytest.mark.parametrize("step", [0, -1, float("nan"), float("inf"), .001])
def test_invalid_or_excessive_levels(step):
    scene = plane()
    with pytest.raises(ProjectError):
        contour_lines(scene.terrain, terrain_heights(scene), step)
