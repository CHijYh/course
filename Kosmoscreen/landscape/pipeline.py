from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import platform
import time

import numpy as np

from .geodata import load_inputs, texture_image
from .mesh import Mesh, require_valid, write_stl
from .worldcover import load_worldcover
from .objects import build_objects, read_features
from .project import MAX_GRID_NODES, MIN_GRID_NODES, Project
from .terrain import sample_surface, terrain_mesh
from .union import unite
from .contours import elevation_metadata


@dataclass
class Scene:
    terrain: Mesh
    objects: list
    texture: np.ndarray
    dimensions: tuple
    elevation: dict | None = None


def file_hash(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            sha.update(block)
    return sha.hexdigest()


def validate_data(project):
    crs, bounds, dem, rgb = load_inputs(project)
    surface = sample_surface(dem, crs, bounds, project.parameters)
    texture_image(rgb, crs, bounds)
    cover, source = load_worldcover(project, crs, bounds)
    _, objects = build_objects(read_features(project.file("objects")), surface, crs, project.parameters, cover)
    long_side_m = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    pixel_m = max(dem.metadata["pixel_size_m"])
    suggested_nodes = None
    if 0 < pixel_m < math.inf:
        intervals = min(long_side_m / pixel_m, MAX_GRID_NODES - 1)
        suggested_nodes = max(MIN_GRID_NODES, round(intervals) + 1)
    return {**({"worldcover": source} if source else {}),
            "valid": True, "crs": str(crs), "bounds_m": bounds, "dem": dem.metadata, "rgb": rgb.metadata,
            "grid_shape": list(surface.z.shape), "grid_step_m": long_side_m / (project.parameters.grid_nodes - 1),
            "suggested_grid_nodes": suggested_nodes, "objects": objects,
            "note": "Проверены файлы, покрытия, параметры и объекты; объединение ещё не выполнено."}


def build_model(project: Project | str | Path, progress=lambda message: None, *, preview=False):
    if not isinstance(project, Project):
        project = Project.load(project)
    project.validate()
    start = time.perf_counter()
    last = start
    report = {"status": "building", "units": "millimetres", "created_at": datetime.now(timezone.utc).isoformat(),
              "project": project.path.name, "title": project.data.get("title", project.path.stem),
              "dataset_kind": project.data.get("dataset_kind", "unspecified"),
              "sources": project.data.get("sources", []), "parameters": asdict(project.parameters),
              "timings_seconds": {}, "slicer": {"status": "not_verified"},
              "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                          **{lib: importlib.metadata.version(lib) for lib in ("numpy", "rasterio", "manifold3d", "pyvista")}}}
    stage = "load"

    def finish(name):
        nonlocal last
        now = time.perf_counter()
        report["timings_seconds"][name] = round(now-last, 6)
        last = now

    try:
        progress("Этап 1/8: загрузка и совмещение геоданных")
        crs, bounds, dem, rgb = load_inputs(project)
        features = read_features(project.file("objects"))
        report.update(working_crs=str(crs), bounds_m=list(bounds),
                      real_dimensions_m=[bounds[2]-bounds[0], bounds[3]-bounds[1]], dem=dem.metadata, rgb=rgb.metadata)
        report["input_hashes"] = {key: {"file": project.data["inputs"][key], "sha256": file_hash(project.file(key))}
                                  for key in ("dem", "rgb", "objects") if project.file(key)}
        finish("load")
        stage = "heights"
        progress("Этапы 2–3/8: пропуски высот, интерполяция и масштабирование")
        surface = sample_surface(dem, crs, bounds, project.parameters)
        report.update(grid_shape=list(surface.z.shape), scale_mm_per_m=surface.scale,
                      elevation_range_m=[surface.min_elevation, surface.max_elevation])
        finish("heights")
        stage = "terrain"
        progress("Этап 4/8: замкнутый рельеф, стенки и дно")
        terrain = terrain_mesh(surface)
        report["terrain_validation"] = require_valid(terrain)
        finish("terrain")
        stage = "objects"
        progress("Этапы 5–6/8: здания, стволы и кроны")
        cover, source = load_worldcover(project, crs, bounds)
        objects, records = build_objects(features, surface, crs, project.parameters, cover)
        if source:
            if project.file("worldcover"):
                report["input_hashes"]["worldcover"] = {"file": source["file"], "sha256": file_hash(project.file("worldcover"))}
            source["trees_added"] = sum(r.get("placement") == "worldcover" for r in records)
            report["worldcover"] = source
        report["objects"] = records
        report["object_counts"] = {
            "buildings_added": sum(r["kind"] == "building" and r["status"] == "added" for r in records),
            "trees_added": sum(r["kind"] == "tree" and r["status"] == "added" for r in records),
            "skipped": sum(r["status"] == "skipped" for r in records)}
        finish("objects")
        united = None
        if not preview:
            stage = "union"
            progress("Этап 7/8: геометрическое объединение manifold3d")
            united, union_report = unite([terrain] + [mesh for _, mesh in objects], progress)
            report["union"] = union_report
            finish("union")
            stage = "validation"
            progress("Этап 8/8: проверка внешней поверхности")
            report["validation"] = require_valid(united)
            finish("validation")
        stage = "texture"
        texture = texture_image(rgb, crs, bounds)
        finish("texture")
        report.update(status="scene_ready" if preview else "geometry_validated",
                      total_build_seconds=round(time.perf_counter()-start, 6))
        scene = Scene(terrain, objects, texture, (float(surface.xs[-1]), float(surface.ys[-1])), elevation_metadata(report))
        return scene, united, report
    except Exception as exc:
        report.update(status="failed", failed_stage=stage, error=str(exc), total_build_seconds=time.perf_counter()-start)
        output = project.output / "preview" if preview else project.output
        output.mkdir(parents=True, exist_ok=True)
        save_report(report, output / "failure_report.json")
        save_report(report, output / "report.json")
        raise


def save_scene(scene, united, destination):
    arrays = {"terrain_v": scene.terrain.vertices, "terrain_f": scene.terrain.faces,
              "texture": scene.texture,
              "dimensions": np.array(scene.dimensions)}
    if united is not None:
        arrays.update(united_v=united.vertices, united_f=united.faces)
    if scene.elevation is not None:
        arrays["elevation"] = np.array(json.dumps(scene.elevation))
    metadata = []
    for i, (kind, mesh) in enumerate(scene.objects):
        arrays[f"object_{i}_v"], arrays[f"object_{i}_f"] = mesh.vertices, mesh.faces
        metadata.append({"kind": kind, "name": mesh.name})
    arrays["metadata"] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(destination, **arrays)


def load_scene(path):
    with np.load(path, allow_pickle=False) as archive:
        objects = [(item["kind"], Mesh(archive[f"object_{i}_v"], archive[f"object_{i}_f"], item["name"]))
                   for i, item in enumerate(json.loads(str(archive["metadata"])))]
        scene = Scene(Mesh(archive["terrain_v"], archive["terrain_f"], "terrain"), objects,
                      archive["texture"], tuple(archive["dimensions"]),
                      json.loads(str(archive["elevation"])) if "elevation" in archive else None)
        united = Mesh(archive["united_v"], archive["united_f"], "united") if "united_v" in archive else None
        return scene, united


def save_preview(scene, report, output):
    output = Path(output) / "preview"
    output.mkdir(parents=True, exist_ok=True)
    save_scene(scene, None, output / "scene.npz")
    save_report(report, output / "report.json")
    return output / "scene.npz"


def prepare_print(scene, report, progress=lambda message: None):
    """Подготовить ранее построенную сцену к экспорту по запросу пользователя."""
    started = time.perf_counter()
    progress("Подготовка STL: объединение объектов")
    united, union_report = unite([scene.terrain] + [mesh for _, mesh in scene.objects], progress)
    validation = require_valid(united)
    report.update(union=union_report, validation=validation, status="geometry_validated")
    report["timings_seconds"]["prepare_print"] = round(time.perf_counter()-started, 6)
    return united


def save_report(report, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_result(scene, mesh, report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        report["stl_validation"] = write_stl(mesh, output / "model.stl")
        report["stl_bytes"] = (output / "model.stl").stat().st_size
        report["stl_sha256"] = file_hash(output / "model.stl")
        report["timings_seconds"]["export"] = round(time.perf_counter()-started, 6)
        save_scene(scene, mesh, output / "scene.npz")
        report["status"] = "export_validated"
        save_report(report, output / "report.json")
        lines = [f"# {report['title']}", "", f"Набор данных: **{report['dataset_kind']}**.", "",
                 "STL записан в миллиметрах и проверен после повторного чтения.",
                 "Геометрическая проверка выполнена; статус слайсера: **не проверен**.", "",
                 f"Размеры, мм: {report['validation']['dimensions_mm']}",
                 f"Треугольники: {report['validation']['triangles']}; связных тел: {report['validation']['components']}.",
                 f"Объекты: {report['object_counts']}", "", "## Время этапов", "", "| Этап | Секунды |", "|---|---:|"]
        lines.extend(f"| {k} | {v:.3f} |" for k, v in report["timings_seconds"].items())
        lines.extend(["", "## Объекты и происхождение", ""])
        for obj in report["objects"]:
            detail = obj.get("reason", obj.get("height_rule", ""))
            lines.append(f"- {obj['id']}: {obj['status']}; {obj.get('source', '')}; {detail}")
            if obj.get("enlarged"):
                lines.append(f"  Укрупнение для печати (мм): {obj['enlarged']}")
        (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    except Exception as exc:
        report.update(status="failed", failed_stage="export", error=str(exc))
        save_report(report, output / "failure_report.json")
        save_report(report, output / "report.json")
        raise

