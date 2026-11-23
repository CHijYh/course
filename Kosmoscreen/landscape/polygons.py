"""Геометрические операции с простыми многоугольниками и триангуляция методом отсечения ушей."""
from __future__ import annotations

import math

from .project import ProjectError

EPS = 1e-8


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def on_segment(p, a, b):
    return (abs(cross(a, b, p)) <= EPS and min(a[0], b[0])-EPS <= p[0] <= max(a[0], b[0])+EPS
            and min(a[1], b[1])-EPS <= p[1] <= max(a[1], b[1])+EPS)


def intersection(a, b, c, d):
    den = (b[0]-a[0])*(d[1]-c[1]) - (b[1]-a[1])*(d[0]-c[0])
    if abs(den) < EPS:
        return None
    t = ((c[0]-a[0])*(d[1]-c[1]) - (c[1]-a[1])*(d[0]-c[0])) / den
    u = ((c[0]-a[0])*(b[1]-a[1]) - (c[1]-a[1])*(b[0]-a[0])) / den
    if -EPS <= t <= 1+EPS and -EPS <= u <= 1+EPS:
        return (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))
    return None


def segments_intersect(a, b, c, d):
    return intersection(a, b, c, d) is not None or any((on_segment(a, c, d), on_segment(b, c, d),
                                                                     on_segment(c, a, b), on_segment(d, a, b)))


def area(poly):
    return sum(a[0]*b[1] - b[0]*a[1] for a, b in zip(poly, poly[1:] + poly[:1])) / 2


def clean_polygon(points, *, require_closed=True):
    p = [tuple(map(float, v[:2])) for v in points]
    if not p or not all(math.isfinite(c) for v in p for c in v):
        raise ProjectError("Контур содержит неконечные координаты.")
    if require_closed and math.dist(p[0], p[-1]) > EPS:
        raise ProjectError("Контур GeoJSON не замкнут.")
    if len(p) > 1 and math.dist(p[0], p[-1]) <= EPS:
        p.pop()
    result = []
    for point in p:
        if not result or math.dist(point, result[-1]) > EPS:
            result.append(point)
    changed = True
    while changed and len(result) > 3:
        changed = False
        for i in range(len(result)):
            a, b, c = result[i-1], result[i], result[(i+1) % len(result)]
            if on_segment(b, a, c):
                result.pop(i)
                changed = True
                break
    if len(result) < 3:
        raise ProjectError("В контуре менее трёх разных точек.")
    if len(result) > 2000:
        raise ProjectError("Контур слишком подробный: более 2000 точек.")
    n = len(result)
    for i in range(n):
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            if segments_intersect(result[i], result[(i+1) % n], result[j], result[(j+1) % n]):
                raise ProjectError("Самопересечение или касание несмежных сторон контура.")
    if abs(area(result)) <= EPS:
        raise ProjectError("Нулевая площадь контура.")
    return result if area(result) > 0 else list(reversed(result))


def in_triangle(p, a, b, c):
    return cross(a, b, p) >= -EPS and cross(b, c, p) >= -EPS and cross(c, a, p) >= -EPS


def triangulate(poly):
    remaining = list(range(len(poly)))
    faces = []
    while len(remaining) > 3:
        for k, b in enumerate(remaining):
            a, c = remaining[k-1], remaining[(k+1) % len(remaining)]
            if cross(poly[a], poly[b], poly[c]) <= EPS:
                continue
            if any(in_triangle(poly[j], poly[a], poly[b], poly[c]) for j in remaining if j not in (a, b, c)):
                continue
            faces.append((a, b, c))
            remaining.pop(k)
            break
        else:
            raise ProjectError("Не удалось триангулировать контур методом отсечения ушей.")
    faces.append(tuple(remaining))
    return faces


def contains(point, poly):
    inside = False
    x, y = point
    for a, b in zip(poly, poly[1:] + poly[:1]):
        if on_segment(point, a, b):
            return True
        if (a[1] > y) != (b[1] > y) and x < a[0] + (y-a[1])*(b[0]-a[0])/(b[1]-a[1]):
            inside = not inside
    return inside


def distance_to_polygon(point, poly):
    if contains(point, poly):
        return 0.0
    def edge_distance(a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        t = max(0, min(1, ((point[0]-a[0])*dx + (point[1]-a[1])*dy)/(dx*dx+dy*dy)))
        return math.hypot(point[0]-a[0]-t*dx, point[1]-a[1]-t*dy)
    return min(edge_distance(a, b) for a, b in zip(poly, poly[1:] + poly[:1]))
