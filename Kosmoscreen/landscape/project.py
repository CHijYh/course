from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os


MIN_GRID_NODES = 3
MAX_GRID_NODES = 1001


class ProjectError(ValueError):
    """Ошибка, которую можно исправить в исходных данных или в параметрах модели."""


@dataclass(frozen=True)
class Parameters:
    long_side_mm: float = 180.0
    base_mm: float = 3.0
    exaggeration: float = 1.0
    grid_nodes: int = 201
    smooth: bool = False
    buildings: bool = True
    trees: bool = True
    adapt_trees: bool = True
    max_buildings: int = 30
    max_trees: int = 30
    seed: int = 42
    tree_spacing_m: float = 8.0
    min_trunk_mm: float = 1.2
    min_crown_mm: float = 3.0
    min_tree_height_mm: float = 4.0
    worldcover: bool = False

    def validate(self):
        for key in ("long_side_mm", "base_mm", "exaggeration", "tree_spacing_m",
                    "min_trunk_mm", "min_crown_mm", "min_tree_height_mm"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ProjectError(f"Параметр {key} должен быть конечным положительным числом.")
        for key, low, high in (("grid_nodes", MIN_GRID_NODES, MAX_GRID_NODES), ("max_buildings", 0, 10000),
                               ("max_trees", 0, 10000), ("seed", 0, 2**32 - 1)):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ProjectError(f"Параметр {key}: допустимо целое число {low}–{high}.")
        for key in ("smooth", "buildings", "trees", "adapt_trees", "worldcover"):
            if type(getattr(self, key)) is not bool:
                raise ProjectError(f"Параметр {key} должен быть true или false.")


@dataclass
class Project:
    path: Path
    data: dict
    parameters: Parameters

    @classmethod
    def load(cls, path: str | Path) -> Project:
        path = Path(path).resolve()
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ProjectError("Требуется объект проекта с version: 1.")
            p = cls(path, data, Parameters(**data.get("parameters", {})))
            p.validate()
            return p
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise ProjectError(f"Не удалось прочитать проект {path}: {exc}") from exc

    def file(self, key: str) -> Path | None:
        value = self.data.get("inputs", {}).get(key)
        return (self.path.parent / value).resolve() if value else None

    @property
    def output(self) -> Path:
        return (self.path.parent / self.data.get("output", "output")).resolve()

    def validate(self):
        self.parameters.validate()
        if not isinstance(self.data.get("inputs"), dict):
            raise ProjectError("В проекте отсутствует раздел inputs.")
        for key in ("dem", "rgb"):
            if not isinstance(self.data["inputs"].get(key), str):
                raise ProjectError(f"Не задан путь inputs.{key}.")
        paths = ["dem", "rgb", "objects"]
        if self.parameters.worldcover and self.parameters.trees:
            if "worldcover" in self.data["inputs"] and not isinstance(self.data["inputs"]["worldcover"], str):
                raise ProjectError("inputs.worldcover должен содержать путь к GeoTIFF.")
            paths.append("worldcover")
        for key in paths:
            path = self.file(key)
            if path is not None and not path.is_file():
                raise ProjectError(f"Файл {key} не найден: {path}")
        area = self.data.get("area", {})
        if not isinstance(area, dict):
            raise ProjectError("area должен быть объектом.")
        if "bounds" not in area and "center" not in area:
            raise ProjectError("Задайте area.center [долгота, широта] или area.bounds в метрах UTM.")

    def coordinates(self):
        from rasterio.crs import CRS
        from rasterio.warp import transform

        area = self.data["area"]
        try:
            center = area.get("center", [42.730, 43.907])
            lon, lat = map(float, center)
            if not (-180 <= lon <= 180 and -80 <= lat <= 84):
                raise ValueError("центр вне диапазона UTM")
            zone = min(60, int((lon + 180) / 6) + 1)
            crs = CRS.from_user_input(area.get("crs", f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"))
            code = crs.to_epsg()
            if code is None or not (32601 <= code <= 32660 or 32701 <= code <= 32760):
                raise ValueError("требуется метрическая UTM EPSG:326xx/327xx")
            if "bounds" in area:
                bounds = tuple(map(float, area["bounds"]))
                if len(bounds) != 4:
                    raise ValueError("bounds должен содержать четыре числа")
            else:
                x, y = transform("EPSG:4326", crs, [lon], [lat])
                w, h = float(area.get("width_m", 1000)), float(area.get("height_m", 1000))
                bounds = (x[0] - w/2, y[0] - h/2, x[0] + w/2, y[0] + h/2)
            if not all(math.isfinite(v) for v in bounds) or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise ValueError("некорректные границы")
            if max(bounds[2]-bounds[0], bounds[3]-bounds[1]) > 20000:
                raise ValueError("участок более 20 км не поддерживается учебной версией")
            return crs, bounds
        except (ValueError, TypeError) as exc:
            raise ProjectError(f"Неверное описание территории: {exc}") from exc

    def save(self, destination: str | Path | None = None):
        destination = Path(destination or self.path).resolve()
        data = dict(self.data)
        data["parameters"] = asdict(self.parameters)
        data["inputs"] = {k: Path(os.path.relpath(self.file(k), destination.parent)).as_posix()
                          for k in self.data["inputs"] if self.file(k) is not None}
        data["output"] = Path(os.path.relpath(self.output, destination.parent)).as_posix()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.path, self.data = destination, data

