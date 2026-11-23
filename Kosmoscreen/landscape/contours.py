"""Построение изолиний по треугольной сетке рельефа без геометрических фильтров VTK."""
from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np

from .project import ProjectError


def top_faces(mesh):
    triangles = mesh.vertices[mesh.faces]
    return np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])[:, 2] > 0


def elevation_metadata(report):
    return {"minimum_m": report["elevation_range_m"][0],
            "maximum_m": report["elevation_range_m"][1],
            "scale_mm_per_m": report["scale_mm_per_m"],
            "base_mm": report["parameters"]["base_mm"],
            "exaggeration": report["parameters"]["exaggeration"]}


def terrain_heights(scene, path=None):
    metadata = scene.elevation
    if metadata is None and path is not None:
        try:
            metadata = elevation_metadata(json.loads(
                Path(path).with_name("report.json").read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    try:
        low, high, scale, base, boost = (float(metadata[k]) for k in
            ("minimum_m", "maximum_m", "scale_mm_per_m", "base_mm", "exaggeration"))
        if not all(math.isfinite(v) for v in (low, high, scale, base, boost)) or scale <= 0 or boost <= 0 or high < low:
            raise ValueError()
        heights = low + (scene.terrain.vertices[:, 2]-base)/(scale*boost)
        used = scene.terrain.faces[top_faces(scene.terrain)].ravel()
        if not len(used) or not np.allclose([heights[used].min(), heights[used].max()], [low, high], rtol=0, atol=1e-5):
            raise ValueError()
        return heights
    except (KeyError, TypeError, ValueError):
        raise ProjectError("Для изолиний нужны высоты в метрах. Постройте сцену заново: сведения о масштабе отсутствуют или не соответствуют рельефу.") from None


@dataclass
class Contour:
    elevation_m: float
    segments: np.ndarray


def contour_lines(mesh, heights, interval=10.0):
    """Пересечь верхние грани плоскостями, заданными через равный шаг высоты на местности.

    На горизонтальном треугольнике изолиния не определяется однозначно. Общее ребро на уровне
    сечения учитывается один раз. Касания в одной точке, боковые стенки и дно пропускаются.
    """
    if not math.isfinite(interval) or interval <= 0:
        raise ProjectError("Шаг изолиний должен быть положительным конечным числом.")
    heights = np.asarray(heights, dtype=float)
    if heights.shape != (len(mesh.vertices),) or not np.isfinite(heights).all():
        raise ProjectError("Некорректные высоты для изолиний.")
    faces = mesh.faces[top_faces(mesh)]
    if not len(faces):
        return []
    values = heights[faces]
    first, last = math.ceil(values.min()/interval), math.floor(values.max()/interval)
    if last-first+1 > 500:
        raise ProjectError("Получается более 500 уровней изолиний. Увеличьте шаг.")
    minimum, maximum = values.min(axis=1), values.max(axis=1)
    result = []
    for number in range(first, last+1):
        level = number*interval
        candidates = np.flatnonzero((minimum <= level+1e-8) & (maximum >= level-1e-8) & (maximum-minimum > 1e-8))
        segments = {}
        for index in candidates:
            triangle, delta = mesh.vertices[faces[index]], values[index]-level
            delta = np.where(abs(delta) <= 1e-8, 0.0, delta)
            points = {}
            for a, b in ((0, 1), (1, 2), (2, 0)):
                if delta[a] == 0:
                    points[tuple(np.round(triangle[a], 10))] = triangle[a]
                if delta[a]*delta[b] < 0:
                    point = triangle[a] + (triangle[b]-triangle[a]) * (-delta[a]/(delta[b]-delta[a]))
                    points[tuple(np.round(point, 10))] = point
            if len(points) == 2:
                keys = tuple(sorted(points))
                segments[keys] = [points[k] for k in keys]
        if segments:
            result.append(Contour(float(level), np.array(list(segments.values()))))
    return result
