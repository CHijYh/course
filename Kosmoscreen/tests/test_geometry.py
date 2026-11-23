import math
import struct

import numpy as np
import pytest

from landscape.geodata import bilinear, fill_nodata, smooth_once
from landscape.mesh import Mesh, check_mesh, read_stl, require_valid, write_stl
from landscape.objects import building_height, cylinder, ellipsoid, prism
from landscape.polygons import area, clean_polygon, triangulate
from landscape.project import ProjectError
from landscape.terrain import Surface, terrain_mesh
from landscape.union import as_manifold, unite


def flat_surface(z=None):
    if z is None:
        z = np.full((5, 5), 3.)
    return Surface(np.linspace(0, 20, z.shape[1]), np.linspace(0, 20, z.shape[0]), z, .18,
                   (0, 0, 1000, 1000), 0, 0)


def test_four_known_heights():
    values = np.array([[0., 10.], [20., 40.]])
    assert bilinear(values, .25, .5) == pytest.approx(13.75)
    assert bilinear(values, 1., 1.) == 40
    with pytest.raises(ProjectError, match="не покрывает"):
        bilinear(values, -.5, 0.)


def test_exact_nearest_and_five_percent_limit():
    values = np.arange(100, dtype=float).reshape(10, 10)
    mask = np.zeros((10, 10), bool)
    mask[3, 3:8] = True
    filled, report = fill_nodata(values, mask, np.ones_like(mask))
    assert report["missing_fraction"] == .05
    assert filled[3, 5] == 25  # При равных расстояниях берётся первый исходный пиксель в порядке обхода строк.
    mask[8, 8] = True
    with pytest.raises(ProjectError, match="5%"):
        fill_nodata(values, mask, np.ones_like(mask))


def test_smoothing_preserves_boundary():
    values = np.arange(25, dtype=float).reshape(5, 5)
    values[2, 2] = 100
    smooth = smooth_once(values)
    assert np.array_equal(smooth[[0, -1]], values[[0, -1]])
    assert np.array_equal(smooth[:, [0, -1]], values[:, [0, -1]])
    assert smooth[2, 2] == pytest.approx(values[1:4, 1:4].mean())


def test_constant_terrain_and_bottom_volume():
    mesh = terrain_mesh(flat_surface())
    report = require_valid(mesh)
    assert report["volume_mm3"] == pytest.approx(20*20*3)
    assert report["dimensions_mm"] == [20., 20., 3.]
    assert len(as_manifold(mesh).decompose()) == 1


def test_slope_direction_and_triangle_interpolation():
    x, y = np.meshgrid(np.linspace(0, 20, 5), np.linspace(0, 20, 5))
    surface = flat_surface(3 + .2*x + .1*y)
    assert surface.height(7, 12) == pytest.approx(5.6)
    assert require_valid(terrain_mesh(surface))["volume_mm3"] == pytest.approx(2400)
    folded = flat_surface(np.array([[3., 3.], [3., 13.]]))
    assert folded.height(10, 10) == pytest.approx(8)
    assert folded.height(15, 5) == pytest.approx(5.5)


@pytest.mark.parametrize("ring", [
    [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
    [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10), (0, 0)],
    [(0, 0), (0, 10), (4, 10), (4, 4), (10, 4), (10, 0), (0, 0)],
])
def test_ear_clipping_area_and_prism(ring):
    poly = clean_polygon(ring)
    triangles = triangulate(poly)
    assert len(triangles) == len(poly)-2
    assert sum(area([poly[i] for i in f]) for f in triangles) == pytest.approx(area(poly))
    mesh = prism(poly, 0, 5, "building")
    assert require_valid(mesh)["volume_mm3"] == pytest.approx(area(poly)*5)


def test_reject_crossing_open_and_collinear_rings():
    with pytest.raises(ProjectError):
        clean_polygon([(0, 0), (10, 10), (0, 10), (10, 0), (0, 0)])
    with pytest.raises(ProjectError, match="не замкнут"):
        clean_polygon([(0, 0), (10, 0), (0, 10)])
    with pytest.raises(ProjectError):
        clean_polygon([(0, 0), (1, 0), (2, 0), (0, 0)])
    assert len(clean_polygon([(0, 0), (0, 0), (5, 0), (10, 0), (10, 10), (0, 10), (0, 0)])) == 4


def test_building_height_fallback():
    assert building_height({"height": "12 m", "building:levels": "8"}) == (12., "height", False)
    assert building_height({"height": "NaN", "building:levels": "2"})[0] == 6
    assert building_height({"height": -3}) == (6., "default 6 m", True)


def test_roof_covers_interior_peak_and_slope_connects():
    z = np.full((5, 5), 3.)
    z[2, 2] = 12
    surface = flat_surface(z)
    poly = [(4, 4), (16, 4), (16, 16), (4, 16)]
    assert surface.maximum_under(poly) == 12
    building = prism(poly, 1.5, surface.maximum_under(poly)+2, "slope_building")
    united, report = unite([terrain_mesh(surface), building])
    assert report["components"] == 1
    assert require_valid(united)["volume_mm3"] > require_valid(terrain_mesh(surface))["volume_mm3"]


def test_tree_closed_parts_overlap_and_union():
    trunk = cylinder(10, 10, .6, 1.5, 6, "trunk")
    crown = ellipsoid(10, 10, 6, 1.5, 2, "crown")
    require_valid(trunk)
    require_valid(crown)
    assert (as_manifold(trunk) ^ as_manifold(crown)).volume() > 0
    united, _ = unite([terrain_mesh(flat_surface()), trunk, crown])
    assert require_valid(united)["components"] == 1


def test_validator_rejects_holes_orientation_and_disconnected_meshes():
    mesh = terrain_mesh(flat_surface())
    assert not check_mesh(Mesh(mesh.vertices, mesh.faces[:-1]))["valid"]
    wrong = mesh.faces.copy()
    wrong[0] = wrong[0, ::-1]
    assert not check_mesh(Mesh(mesh.vertices, wrong))["valid"]
    joined = Mesh(np.concatenate((mesh.vertices, mesh.vertices+[30, 0, 0])),
                  np.concatenate((mesh.faces, mesh.faces+len(mesh.vertices))))
    assert check_mesh(joined)["components"] == 2
    assert not check_mesh(joined)["valid"]


def test_binary_stl_roundtrip(tmp_path):
    mesh = terrain_mesh(flat_surface())
    path = tmp_path/"тест.stl"
    result = write_stl(mesh, path)
    raw = path.read_bytes()
    assert len(raw) == 84+50*len(mesh.faces)
    assert struct.unpack_from("<I", raw, 80)[0] == len(mesh.faces)
    assert result["valid"]
    assert require_valid(read_stl(path))["volume_mm3"] == pytest.approx(1200)
