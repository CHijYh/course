from dataclasses import replace
from pathlib import Path
import copy
import json

import numpy as np
import pytest
import rasterio
from rasterio.warp import transform

from landscape.demo import create_demo
from landscape.geodata import load_inputs
from landscape.mesh import read_stl, require_valid
from landscape.objects import build_objects, read_features
from landscape.pipeline import build_model, load_scene, save_result, validate_data
from landscape.project import Parameters, Project, ProjectError
from landscape.terrain import sample_surface


@pytest.fixture
def project(tmp_path):
    p = Project.load(create_demo(tmp_path/"исходные данные", method="analytic-legacy"))
    p.parameters = replace(p.parameters, grid_nodes=31, max_trees=5)
    return p


def test_complete_pipeline_and_relative_project_move(project, tmp_path):
    scene, mesh, report = build_model(project)
    assert report["object_counts"] == {"buildings_added": 8, "trees_added": 5, "skipped": 0}
    assert report["dem"]["missing_inside"] == 1
    assert report["validation"]["components"] == 1
    assert report["validation"]["dimensions_mm"][:2] == pytest.approx([180, 180])
    save_result(scene, mesh, report, project.output)
    actual = read_stl(project.output/"model.stl")
    assert require_valid(actual)["valid"]
    restored, united = load_scene(project.output/"scene.npz")
    assert np.array_equal(restored.texture, scene.texture)
    assert len(united.faces) == len(mesh.faces)
    assert len(restored.objects) == 18
    target = tmp_path/"перенос"/"project.json"
    project.save(target)
    again = Project.load(target)
    assert again.file("dem").is_file()
    assert not Path(again.data["inputs"]["dem"]).is_absolute()
    assert again.parameters.grid_nodes == 31


def test_seeded_objects_and_mesh_reproducible(project):
    first = build_model(project)
    second = build_model(project)
    assert first[2]["objects"] == second[2]["objects"]
    assert np.array_equal(first[1].vertices, second[1].vertices)
    assert np.array_equal(first[1].faces, second[1].faces)
    generated = [r for r in first[2]["objects"] if r.get("generated")]
    assert len(generated) == 1
    assert generated[0]["enlarged"]["trunk_diameter"]["to_mm"] == 1.2


def test_skip_invalid_buildings_and_tree_collisions(project):
    crs, bounds, dem, rgb = load_inputs(project)
    surface = sample_surface(dem, crs, bounds, project.parameters)
    features = read_features(project.file("objects"))[:1]
    building = features[0]
    hole = copy.deepcopy(building)
    hole["id"] = "hole"
    hole["geometry"]["coordinates"].append(hole["geometry"]["coordinates"][0])
    collision = {"type": "Feature", "id": "collision", "properties": {"natural": "tree"},
                 "geometry": {"type": "Point", "coordinates": building["geometry"]["coordinates"][0][0]}}
    outside = copy.deepcopy(collision)
    outside["id"] = "outside"
    lon, lat = transform(crs, "EPSG:4326", [bounds[0]+1], [bounds[1]+1])
    outside["geometry"]["coordinates"] = [lon[0], lat[0]]
    _, records = build_objects(features+[hole, collision, outside], surface, crs, project.parameters)
    by_id = {r["id"]: r for r in records}
    assert "отверстиями" in by_id["hole"]["reason"]
    assert "пересекают здание" in by_id["collision"]["reason"]
    assert "границу" in by_id["outside"]["reason"]


def test_reject_missing_crs_and_excessive_nodata(project):
    with rasterio.open(project.file("dem"), "r+") as src:
        values = src.read(1)
        values[10:25, 10:25] = src.nodata
        src.write(values, 1)
    with pytest.raises(ProjectError, match="5%"):
        build_model(project)
    failure = json.loads((project.output/"failure_report.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert not (project.output/"model.stl").exists()
    rgb = project.file("rgb")
    with rasterio.open(rgb) as src:
        values, profile = src.read(), src.profile
    profile.pop("crs")
    with rasterio.open(rgb, "w", **profile) as out:
        out.write(values)
    # Восстанавливаем DEM, чтобы отдельно проверить обработку RGB-растра.
    create_demo(project.path.parent, method="analytic-legacy")
    with rasterio.open(rgb, "w", **profile) as out:
        out.write(values)
    with pytest.raises(ProjectError, match="отсутствует CRS"):
        validate_data(project)


def test_outside_coverage_and_invalid_crs(project):
    project.data["area"]["center"] = [42.8, 43.907]
    with pytest.raises(ProjectError):
        validate_data(project)
    project.data["area"]["crs"] = "EPSG:4326"
    with pytest.raises(ProjectError, match="UTM"):
        project.coordinates()


def test_rgb_zero_channel_is_not_missing_pixel(project):
    with rasterio.open(project.file("rgb"), "r+") as src:
        values = src.read()
        values[2, 50:55, 50:55] = 0
        src.nodata = 0
        src.write(values)
    assert validate_data(project)["valid"]


def test_terrain_only_and_rectangular_model(project):
    project.parameters = replace(project.parameters, buildings=False, trees=False)
    project.data["area"]["height_m"] = 500
    scene, mesh, report = build_model(project)
    assert report["grid_shape"] == [16, 31]
    assert require_valid(mesh)["dimensions_mm"][:2] == pytest.approx([180, 90])
    assert not scene.objects


def test_exaggeration_does_not_scale_building_height(project):
    scene, _, report = build_model(project)
    project.parameters = replace(project.parameters, exaggeration=2)
    _, _, boosted = build_model(project)
    for a, b in zip(report["objects"], boosted["objects"]):
        if a["kind"] == "building":
            h = a["height_m"]*report["scale_mm_per_m"]
            assert b["roof_mm"] == pytest.approx(3+2*(a["roof_mm"]-h-3)+h)


@pytest.mark.parametrize("values", [{"grid_nodes": 2}, {"grid_nodes": 20.5}, {"base_mm": -1},
                                   {"exaggeration": float("nan")}, {"smooth": "false"}, {"max_trees": 10001},
                                   {"max_buildings": -1}, {"max_trees": 2.5}, {"max_buildings": True}])
def test_invalid_parameters(values):
    with pytest.raises(ProjectError):
        Parameters(**values).validate()


def test_large_scene_without_union_preserves_print_result(project, monkeypatch):
    from landscape.pipeline import save_preview
    project.parameters = replace(project.parameters, max_buildings=500, max_trees=100, adapt_trees=False)
    # Повторяем корректные контуры с разными идентификаторами, чтобы проверить ограничение
    # числа зданий в режиме просмотра без объединения совпадающих моделей.
    path = project.file("objects")
    data = json.loads(path.read_text(encoding="utf-8"))
    original = copy.deepcopy(data["features"][0])
    for i in range(35):
        feature = copy.deepcopy(original)
        feature["id"] = f"additional_{i}"
        data["features"].append(feature)
    path.write_text(json.dumps(data), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview must not perform boolean union")
    monkeypatch.setattr("landscape.pipeline.unite", forbidden)
    project.output.mkdir(parents=True, exist_ok=True)
    sentinel = project.output / "model.stl"
    sentinel.write_bytes(b"previous print artifact")
    scene, mesh, report = build_model(project, preview=True)
    assert mesh is None and report["status"] == "scene_ready"
    assert report["object_counts"]["buildings_added"] == 43
    assert report["object_counts"]["trees_added"] == 100
    assert "union" not in report and "validation" not in report
    destination = save_preview(scene, report, project.output)
    restored, united = load_scene(destination)
    assert united is None and len(restored.objects) == 243
    assert sentinel.read_bytes() == b"previous print artifact"


def test_preview_can_be_prepared_for_print(project):
    from landscape.pipeline import prepare_print
    scene, mesh, report = build_model(project, preview=True)
    mesh = prepare_print(scene, report)
    assert report["status"] == "geometry_validated"
    assert require_valid(mesh)["components"] == 1
    save_result(scene, mesh, report, project.output)
    assert report["status"] == "export_validated"
