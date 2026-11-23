from __future__ import annotations

import json
import math
import random
import re

import numpy as np
from rasterio.warp import transform

from .worldcover import tree_candidates
from .mesh import Mesh
from .polygons import clean_polygon, contains, distance_to_polygon, triangulate
from .project import ProjectError


def positive_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*(?:m|м)?\s*", value)
        if not match:
            return None
        value = match[1].replace(",", ".")
    try:
        value = float(value)
        return value if math.isfinite(value) and 0 < value <= 1000 else None
    except (TypeError, ValueError):
        return None


def building_height(properties):
    value = positive_number(properties.get("height"))
    if value:
        return value, "height", False
    levels = positive_number(properties.get("building:levels"))
    if levels:
        return levels*3, "building:levels × 3 m", True
    return 6.0, "default 6 m", True


def prism(poly, lower, upper, name):
    roof = triangulate(poly)
    n = len(poly)
    vertices = [(x, y, lower) for x, y in poly] + [(x, y, upper) for x, y in poly]
    faces = [(c, b, a) for a, b, c in roof] + [(a+n, b+n, c+n) for a, b, c in roof]
    for i in range(n):
        j = (i+1) % n
        faces.extend(((i, j, j+n), (i, j+n, i+n)))
    return Mesh(vertices, faces, name)


def cylinder(x, y, radius, lower, upper, name, segments=16):
    poly = [(x+radius*math.cos(2*math.pi*i/segments), y+radius*math.sin(2*math.pi*i/segments))
            for i in range(segments)]
    return prism(poly, lower, upper, name)


def ellipsoid(x, y, z, radius, vertical_radius, name, segments=16, rings=8):
    vertices = [(x, y, z-vertical_radius)]
    for j in range(1, rings):
        latitude = -math.pi/2 + j*math.pi/rings
        r = radius*math.cos(latitude)
        for i in range(segments):
            angle = 2*math.pi*i/segments
            vertices.append((x+r*math.cos(angle), y+r*math.sin(angle), z+vertical_radius*math.sin(latitude)))
    top = len(vertices)
    vertices.append((x, y, z+vertical_radius))
    faces = []
    for i in range(segments):
        j = (i+1) % segments
        faces.append((0, 1+j, 1+i))
        for ring in range(rings-2):
            a, b = 1+ring*segments+i, 1+ring*segments+j
            faces.extend(((a, b, b+segments), (a, b+segments, a+segments)))
        faces.append((1+(rings-2)*segments+i, 1+(rings-2)*segments+j, top))
    return Mesh(vertices, faces, name)


def read_features(path):
    if path is None:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
            raise ProjectError("Объекты должны быть GeoJSON FeatureCollection.")
        if "crs" in data:
            name = str(data["crs"])
            if "4326" not in name and "CRS84" not in name:
                raise ProjectError("GeoJSON должен содержать координаты WGS84 [долгота, широта].")
        return data["features"]
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise ProjectError(f"Не удалось прочитать GeoJSON: {exc}") from exc


def build_objects(features, surface, crs, parameters, worldcover=None):
    meshes, records, footprints, forests, points = [], [], [], [], []
    crossing_buildings = []
    scale = surface.scale

    def to_local(coords):
        if not coords or any(len(c) < 2 or not (-180 <= float(c[0]) <= 180 and -90 <= float(c[1]) <= 90) for c in coords):
            raise ProjectError("Неверные координаты WGS84.")
        x, y = transform("EPSG:4326", crs, [c[0] for c in coords], [c[1] for c in coords])
        # Округление до 0,000001 мм (одного нанометра) устраняет погрешность пересчёта координат.
        return [(round((a-surface.bounds[0])*scale, 6), round((b-surface.bounds[1])*scale, 6)) for a, b in zip(x, y)]

    def polygon(geometry):
        if geometry.get("type") != "Polygon":
            raise ProjectError("Поддерживается только простой Polygon.")
        rings = geometry.get("coordinates", [])
        if len(rings) != 1:
            raise ProjectError("Контур с отверстиями или без внешнего кольца.")
        return clean_polygon(to_local(rings[0]))

    def inside(point, radius=0):
        return radius <= point[0] <= surface.xs[-1]-radius and radius <= point[1] <= surface.ys[-1]-radius

    count_buildings = 0
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            records.append({"id": str(index), "status": "skipped", "reason": "Feature не является объектом"})
            continue
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        record = {"id": str(feature.get("id", index)), "source": props.get("source", "unspecified GeoJSON"),
                  "status": "skipped", "kind": "unknown"}
        records.append(record)
        try:
            if props.get("building") and props.get("building") != "no":
                record["kind"] = "building"
                poly = polygon(geom)
                if not all(inside(p) for p in poly):
                    crossing_buildings.append(poly)
                    raise ProjectError("Здание пересекает границу участка.")
                # Даже если здание отключено или выходит за лимит, его контур остаётся закрытым для посадки деревьев.
                footprints.append(poly)
                if not parameters.buildings or count_buildings >= parameters.max_buildings:
                    raise ProjectError("Здания отключены или достигнут лимит.")
                height, rule, estimated = building_height(props)
                roof = surface.maximum_under(poly)+height*scale
                mesh = prism(poly, parameters.base_mm/2, roof, f"building_{record['id']}")
                meshes.append(("building", mesh))
                count_buildings += 1
                record.update(status="added", height_m=height, height_rule=rule, estimated=estimated,
                              roof_mm=roof, footprint_mm=[list(p) for p in poly])
            elif geom.get("type") == "Point" and props.get("natural") == "tree":
                record["kind"] = "tree"
                points.append((to_local([geom["coordinates"]])[0], props, record))
                record["status"] = "pending"
            elif props.get("landuse") == "forest" or props.get("natural") == "wood":
                record["kind"] = "forest"
                forests.append((polygon(geom), record))
                record["status"] = "placement_area"
            else:
                raise ProjectError("Неподдерживаемый тип объекта или отсутствует тег building/natural/landuse.")
        except (ProjectError, TypeError, ValueError, KeyError, IndexError) as exc:
            record.update(status="skipped", reason=str(exc))

    tree_count = 0

    # По ограничивающим прямоугольникам отбрасываем далёкие здания до точного расчёта расстояний.
    def indexed(polygons):
        return [(p, min(x for x, y in p), min(y for x, y in p),
                 max(x for x, y in p), max(y for x, y in p)) for p in polygons]

    obstacles = indexed(footprints)
    # Все соседи в пределах проверяемого расстояния лежат в этой ячейке или в одной из восьми соседних.
    largest_diameter = max((positive_number(props.get(key)) or 0
                            for _, props, _ in points for key in ("diameter_crown", "diameter_trunk")), default=0)
    bucket_size = max(parameters.tree_spacing_m * scale, 5 * scale, largest_diameter * scale,
                      max(parameters.min_trunk_mm, parameters.min_crown_mm) if parameters.adapt_trees else 0)
    buckets = {}

    def add_tree(point, props, record, generated=False):
        nonlocal tree_count
        if not parameters.trees or tree_count >= parameters.max_trees:
            record.update(status="skipped", reason="Деревья отключены или достигнут лимит.")
            return False
        height_value = positive_number(props.get("height"))
        crown_value = positive_number(props.get("diameter_crown"))
        height, crown = height_value or 10.0, crown_value or 5.0
        trunk = positive_number(props.get("diameter_trunk")) or 0.4
        natural = {"height": height*scale, "crown_diameter": crown*scale, "trunk_diameter": trunk*scale}
        actual = dict(natural)
        if parameters.adapt_trees:
            actual["height"] = max(actual["height"], parameters.min_tree_height_mm)
            actual["crown_diameter"] = max(actual["crown_diameter"], parameters.min_crown_mm)
            actual["trunk_diameter"] = max(actual["trunk_diameter"], parameters.min_trunk_mm)
        radius = max(actual["crown_diameter"], actual["trunk_diameter"])/2
        bx, by = (math.floor(v / bucket_size) for v in point)
        x, y = point
        reason = None
        if not inside(point, radius):
            reason = "Крона или ствол выходят за границу макета."
        elif any(left-radius <= x <= right+radius and bottom-radius <= y <= top+radius
                 and distance_to_polygon(point, poly) <= radius for poly, left, bottom, right, top in obstacles):
            reason = "Крона или ствол пересекают здание."
        elif any(math.dist(point, p) < max(parameters.tree_spacing_m*scale, radius+r)
                 for i in range(bx-1, bx+2) for j in range(by-1, by+2) for p, r in buckets.get((i, j), ())):
            reason = "Недостаточное расстояние между деревьями."
        if reason:
            record.update(status="skipped", reason=reason)
            return False
        ground = surface.height(x, y)
        h = actual["height"]
        stem = cylinder(x, y, actual["trunk_diameter"]/2, parameters.base_mm/2, ground+h*0.65,
                        f"tree_{record['id']}_trunk")
        crown_mesh = ellipsoid(x, y, ground+h*0.7, actual["crown_diameter"]/2, h*0.3,
                               f"tree_{record['id']}_crown")
        meshes.extend((("trunk", stem), ("crown", crown_mesh)))
        tree_count += 1
        buckets.setdefault((bx, by), []).append((point, radius))
        record.update(status="added", position_mm=list(point), generated=generated,
                      dimensions_mm=actual, scaled_dimensions_mm=natural,
                      enlarged={k: {"from_mm": natural[k], "to_mm": v} for k, v in actual.items() if v > natural[k]},
                      estimated_height=height_value is None, estimated_crown=crown_value is None)
        return True

    for point, props, record in points:
        add_tree(point, props, record)
    rng = random.Random(parameters.seed)
    for poly, forest in forests:
        if not parameters.trees:
            forest.update(generated_trees=0, attempts=0)
            continue
        x0, x1 = max(0, min(p[0] for p in poly)), min(surface.xs[-1], max(p[0] for p in poly))
        y0, y1 = max(0, min(p[1] for p in poly)), min(surface.ys[-1], max(p[1] for p in poly))
        added, attempts = 0, 0
        if x0 < x1 and y0 < y1:
            while tree_count < parameters.max_trees and attempts < 3000:
                attempts += 1
                p = (rng.uniform(x0, x1), rng.uniform(y0, y1))
                if not contains(p, poly):
                    continue
                record = {"id": f"{forest['id']}_generated_{added+1}", "kind": "tree",
                          "source": f"procedural placement in {forest['id']}; seed={parameters.seed}"}
                if add_tree(p, {}, record, generated=True):
                    records.append(record)
                    added += 1
        forest.update(generated_trees=added, attempts=attempts)
    if worldcover is not None and parameters.trees and tree_count < parameters.max_trees:
        obstacles.extend(indexed(crossing_buildings))
        excluded = footprints + crossing_buildings + [poly for poly, _ in forests]
        for cell, point in tree_candidates(worldcover, excluded, surface, parameters.seed):
            record = {"id": f"wc_cell_{cell}", "kind": "tree", "placement": "worldcover",
                      "source": "ESA WorldCover, class 10"}
            if add_tree(point, {}, record, generated=True):
                records.append(record)
                if tree_count >= parameters.max_trees:
                    break
    return meshes, records

