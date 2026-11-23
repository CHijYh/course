from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geodata import smooth_once
from .mesh import Mesh
from .polygons import contains, intersection
from .project import ProjectError


@dataclass
class Surface:
    xs: np.ndarray
    ys: np.ndarray
    z: np.ndarray
    scale: float
    bounds: tuple
    min_elevation: float
    max_elevation: float

    def height(self, x, y):
        """Вычислить высоту на треугольной грани, на которую попадает точка после триангуляции."""
        if x < -1e-7 or y < -1e-7 or x > self.xs[-1]+1e-7 or y > self.ys[-1]+1e-7:
            raise ProjectError("Объект выходит за границы рельефа.")
        col = min(len(self.xs)-2, max(0, int(x/self.xs[-1]*(len(self.xs)-1))))
        row = min(len(self.ys)-2, max(0, int(y/self.ys[-1]*(len(self.ys)-1))))
        u = (x-self.xs[col])/(self.xs[col+1]-self.xs[col])
        v = (y-self.ys[row])/(self.ys[row+1]-self.ys[row])
        a, b, c, d = self.z[row, col], self.z[row, col+1], self.z[row+1, col+1], self.z[row+1, col]
        return float(a + u*(b-a) + v*(c-b) if v <= u else a + u*(c-d) + v*(d-a))

    def maximum_under(self, poly):
        """Найти точные экстремумы кусочно-линейной поверхности внутри многоугольника.

        Достаточно проверить вершины контура, узлы сетки внутри него и точки пересечения
        сторон контура с рёбрами сетки, включая диагонали ячеек.
        """
        candidates = list(poly)
        xmin, xmax = min(p[0] for p in poly), max(p[0] for p in poly)
        ymin, ymax = min(p[1] for p in poly), max(p[1] for p in poly)
        c0 = max(0, int(np.searchsorted(self.xs, xmin))-1)
        c1 = min(len(self.xs)-1, int(np.searchsorted(self.xs, xmax)))
        r0 = max(0, int(np.searchsorted(self.ys, ymin))-1)
        r1 = min(len(self.ys)-1, int(np.searchsorted(self.ys, ymax)))
        sides = list(zip(poly, poly[1:]+poly[:1]))
        edges = set()
        for r in range(r0, r1+1):
            for c in range(c0, c1+1):
                point = (float(self.xs[c]), float(self.ys[r]))
                if contains(point, poly):
                    candidates.append(point)
                if c < c1:
                    edges.add((point, (float(self.xs[c+1]), point[1])))
                if r < r1:
                    edges.add((point, (point[0], float(self.ys[r+1]))))
                if c < c1 and r < r1:
                    edges.add((point, (float(self.xs[c+1]), float(self.ys[r+1]))))
        for a, b in edges:
            for c, d in sides:
                p = intersection(a, b, c, d)
                if p is not None:
                    candidates.append(p)
        return max(self.height(x, y) for x, y in candidates)


def sample_surface(dem, crs, bounds, parameters):
    w, h = bounds[2]-bounds[0], bounds[3]-bounds[1]
    n = parameters.grid_nodes
    nx, ny = max(2, round((n-1)*w/max(w, h))+1), max(2, round((n-1)*h/max(w, h))+1)
    x, y = np.meshgrid(np.linspace(bounds[0], bounds[2], nx), np.linspace(bounds[1], bounds[3], ny))
    heights = dem.sample(x, y, crs)
    if parameters.smooth:
        heights = smooth_once(heights)
    low, high = float(heights.min()), float(heights.max())
    scale = parameters.long_side_mm/max(w, h)
    z = parameters.base_mm + scale*parameters.exaggeration*(heights-low)
    return Surface(np.linspace(0, w*scale, nx), np.linspace(0, h*scale, ny), z, scale, bounds, low, high)


def terrain_mesh(surface: Surface):
    ny, nx = surface.z.shape
    xx, yy = np.meshgrid(surface.xs, surface.ys)
    vertices = list(zip(xx.ravel(), yy.ravel(), surface.z.ravel()))
    faces = []
    for j in range(ny-1):
        for i in range(nx-1):
            a = j*nx+i
            faces.extend(((a, a+1, a+nx+1), (a, a+nx+1, a+nx)))
    # Обходим периметр против часовой стрелки. Вершины дна получают те же координаты X и Y.
    perimeter = (list(range(nx)) + [j*nx+nx-1 for j in range(1, ny)]
                 + [(ny-1)*nx+i for i in range(nx-2, -1, -1)]
                 + [j*nx for j in range(ny-2, 0, -1)])
    bottom = len(vertices)
    vertices.extend((vertices[i][0], vertices[i][1], 0) for i in perimeter)
    center = len(vertices)
    vertices.append((surface.xs[-1]/2, surface.ys[-1]/2, 0))
    for k, a in enumerate(perimeter):
        nxt = (k+1) % len(perimeter)
        b, ba, bb = perimeter[nxt], bottom+k, bottom+nxt
        faces.extend(((a, ba, bb), (a, bb, b), (center, bb, ba)))
    return Mesh(vertices, faces, "terrain")
