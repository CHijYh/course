from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import copy
import csv
import json

from .pipeline import build_model, save_result
from .project import Project


def run(project, progress=print):
    destination = project.output.parent/"experiments"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for grid, adapted in ((101, True), (201, True), (301, True), (201, False)):
        name = f"grid_{grid}_{'adapted' if adapted else 'natural'}"
        progress(f"Эксперимент: {name}")
        params = replace(project.parameters, grid_nodes=grid, adapt_trees=adapted)
        item = Project(project.path, copy.deepcopy(project.data), params)
        scene, mesh, report = build_model(item, progress)
        save_result(scene, mesh, report, destination/name)
        row = {"grid": grid, "adapt_trees": adapted, "terrain_seconds": report["timings_seconds"]["terrain"],
               "heights_seconds": report["timings_seconds"]["heights"], "union_seconds": report["timings_seconds"]["union"],
               "total_build_seconds": report["total_build_seconds"], "triangles": len(mesh.faces),
               "stl_bytes": report["stl_bytes"], **report["object_counts"]}
        rows.append(row)
    with (destination/"results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (destination/"results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# Сравнение параметров", "", f"Проект: {project.path.name}", f"Данные: {project.data.get('dataset_kind', 'unspecified')}", "",
             "Один измеренный запуск каждого варианта. Время включает проверки соответствующего этапа; это не статистика повторных измерений.", "",
             "| Сетка | Адаптация деревьев | Рельеф, с | Объединение, с | Треугольники | STL, байт |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['grid']} | {row['adapt_trees']} | {row['terrain_seconds']:.3f} | {row['union_seconds']:.3f} | {row['triangles']} | {row['stl_bytes']} |")
    lines.extend(["", "Сетки интерполируют один и тот же DEM. Дополнительные узлы не являются новыми измерениями.",
                  "Адаптация меняет размеры и может изменить допустимость размещения деревьев; сравнивайте также object_counts в отчётах.",
                  "Географическая точность без независимых измерений не оценивается."])
    (destination/"results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    progress(f"Таблицы сохранены: {destination}")
    return rows
