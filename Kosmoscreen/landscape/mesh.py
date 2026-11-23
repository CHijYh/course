from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

from .project import ProjectError


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    name: str = "mesh"

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)


def normals(mesh: Mesh):
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.sqrt(np.sum(cross * cross, axis=1))
    return cross / np.maximum(lengths[:, None], 1e-300), lengths / 2


def check_mesh(mesh: Mesh, *, require_single=True) -> dict:
    v, f = mesh.vertices, mesh.faces
    if not len(v) or not len(f) or not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
        raise ProjectError(f"{mesh.name}: пустая сетка, неконечные координаты или неверные индексы.")
    _, areas = normals(mesh)
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    keys = np.sort(edges, axis=1)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    balances = np.bincount(inverse, weights=signs)
    parent = list(range(len(v)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in unique.tolist():
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    components = len({root(i) for i in np.unique(f).tolist()})
    # Сдвиг к началу координат уменьшает потерю точности при вычитании близких больших чисел.
    t = (v - v.mean(axis=0))[f]
    volume = float(np.sum(t[:, 0] * np.cross(t[:, 1], t[:, 2])) / 6)
    report = {"vertices": len(v), "triangles": len(f), "finite": True,
              "degenerate_triangles": int(np.count_nonzero(areas <= 1e-14)),
              "nonmanifold_edges": int(np.count_nonzero(counts != 2)),
              "orientation_errors": int(np.count_nonzero(balances)),
              "components": components, "volume_mm3": volume,
              "bounds_mm": [v.min(axis=0).tolist(), v.max(axis=0).tolist()],
              "dimensions_mm": (v.max(axis=0) - v.min(axis=0)).tolist()}
    report["valid"] = bool(not report["degenerate_triangles"] and not report["nonmanifold_edges"]
                           and not report["orientation_errors"] and volume > 0
                           and (not require_single or components == 1))
    return report


def require_valid(mesh: Mesh):
    report = check_mesh(mesh)
    if not report["valid"]:
        raise ProjectError(f"Некорректная сетка {mesh.name}: {report}")
    return report


def write_stl(mesh: Mesh, destination: str | Path):
    """Записать двоичный STL младшим байтом вперёд и проверить геометрию после преобразования в float32."""
    destination = Path(destination)
    require_valid(mesh)
    # В STL координаты 32-разрядные. Проверяем их с той же точностью, с которой они попадут в файл.
    quantized = Mesh(mesh.vertices.astype(np.float32), mesh.faces, mesh.name + " (float32)")
    require_valid(quantized)
    n, _ = normals(quantized)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(b"Kosmoscreen | coordinates in millimetres | binary STL".ljust(80, b" "))
            stream.write(struct.pack("<I", len(mesh.faces)))
            for normal, triangle in zip(n, quantized.vertices[quantized.faces]):
                stream.write(struct.pack("<12fH", *normal, *triangle.ravel(), 0))
        actual = read_stl(temporary)
        result = require_valid(actual)
        from .union import as_manifold
        body = as_manifold(actual)
        if len(body.decompose()) != 1:
            raise ProjectError("STL после записи содержит несколько тел.")
        result["manifold_status"] = str(body.status())
        temporary.replace(destination)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def read_stl(path: str | Path) -> Mesh:
    """Прочитать двоичный STL и объединить только вершины с точно совпадающими координатами."""
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ProjectError("Неполный заголовок STL.")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ProjectError("Размер бинарного STL не соответствует числу треугольников.")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    triangles = np.frombuffer(raw, dtype=dtype, count=count, offset=84)["vertices"].reshape(-1, 3)
    vertices, inverse = np.unique(triangles, axis=0, return_inverse=True)
    return Mesh(vertices, inverse.reshape(-1, 3), Path(path).stem)

