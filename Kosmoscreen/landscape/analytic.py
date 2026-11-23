"""Поверхность Франке для проверки интерполяции. Значения вычисляются с помощью NumPy.

Формула взята из работ ниже. Программная реализация самостоятельная.
Р. Франке. Интерполяция рассеянных данных: испытания нескольких методов (1982).
https://doi.org/10.1090/S0025-5718-1982-0637296-4
С. Сурьянович, Д. Бингем. Функция Франке, электронная библиотека Университета Саймона Фрейзера.
https://www.sfu.ca/~ssurjano/franke2d.html
Масштабы координат и высот, а также способ построения тестовых геоданных выбраны для проекта.
"""
from __future__ import annotations

import numpy as np


FRANKE_DOI = "https://doi.org/10.1090/S0025-5718-1982-0637296-4"
FRANKE_CATALOG = "https://www.sfu.ca/~ssurjano/franke2d.html"
WIDTH_M = 1000.0
HEIGHT_OFFSET_M = 800.0
HEIGHT_SCALE_M = 120.0


def franke(u, v):
    """Вычислить функцию Франке. Стандартная область проверки — единичный квадрат [0, 1]².

    Для построения запаса DEM по краям участка допустимы конечные координаты за пределами квадрата.
    К ним применяется та же формула, без ограничения координат или периодического переноса.
    """
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise ValueError("Координаты аналитической поверхности должны быть конечными.")
    a, b = 9 * u, 9 * v
    exponents = np.stack((
        -((a - 2)**2 + (b - 2)**2) / 4,
        -(a + 1)**2 / 49 - (b + 1) / 10,
        -((a - 7)**2 + (b - 3)**2) / 4,
        -(a - 4)**2 - (b - 7)**2,
    ), axis=-1)
    return np.sum(np.exp(exponents) * np.array([0.75, 0.75, 0.5, -0.2]), axis=-1)


def analytic_heights(x_m, y_m):
    """Перевести координаты центрального квадрата со стороной 1 км в единичный квадрат Франке и вернуть высоту в метрах."""
    u = np.asarray(x_m, dtype=float) / WIDTH_M + 0.5
    v = np.asarray(y_m, dtype=float) / WIDTH_M + 0.5
    return HEIGHT_OFFSET_M + HEIGHT_SCALE_M * franke(u, v)


def analytic_metadata():
    return {"method": "franke_analytic", "implementation_version": 1,
            "algorithm_references": [
                {"url": FRANKE_DOI, "role": "Published analytical test surface for interpolation"},
                {"url": FRANKE_CATALOG, "role": "Formula and conventional unit-square domain"}],
            "coordinate_mapping": "u=x_m/1000+0.5; v=y_m/1000+0.5; origin at area centre",
            "height_mapping": "h_m=800+120*F(u,v)",
            "area_width_m": WIDTH_M, "height_offset_m": HEIGHT_OFFSET_M,
            "height_scale_m": HEIGHT_SCALE_M, "randomness": "none",
            "halo": "same formula evaluated outside [0,1]^2, no clipping",
            "artificial_nodata_pixel_row_col": [17, 18]}
