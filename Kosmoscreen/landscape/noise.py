"""Двумерный градиентный шум с интерполяцией по полиному пятой степени Перлина (2002).

Алгоритм реализован самостоятельно на NumPy. Для двумерной решётки выбран свой набор
градиентов, а перестановка строится по заданному начальному значению генератора.
Описание метода Перлина: https://doi.org/10.1145/566654.566636
Спектральный синтез описан у Фарра, Якоба и Хамфриса в третьем издании PBRT, раздел 10.6.3:
https://www.pbr-book.org/3ed-2018/Texture/Noise#NoiseIdiomsandSpectralSynthesis
Подробнее об использованных источниках см. в docs/PROCEDURAL_GENERATOR.md.
"""
from __future__ import annotations

import random

import numpy as np


PERLIN_DOI = "https://doi.org/10.1145/566654.566636"
PBRT_NOISE_URL = "https://www.pbr-book.org/3ed-2018/Texture/Noise#NoiseIdiomsandSpectralSynthesis"
BASE_WAVELENGTH_M = 600.0
HEIGHT_ORIGIN_M = 850.0
HEIGHT_AMPLITUDE_M = 180.0
OCTAVES = 4
PERSISTENCE = 0.5
LACUNARITY = 2.0

# Для двумерной решётки выбраны восемь единичных векторов с равным угловым шагом.
_DIAGONAL = 1.0 / np.sqrt(2.0)
_GRADIENTS = np.array([(1, 0), (_DIAGONAL, _DIAGONAL), (0, 1),
                       (-_DIAGONAL, _DIAGONAL), (-1, 0),
                       (-_DIAGONAL, -_DIAGONAL), (0, -1),
                       (_DIAGONAL, -_DIAGONAL)])


def permutation_for_seed(seed: int) -> list[int]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed генератора должен быть целым числом 0–4294967295.")
    permutation = list(range(256))
    random.Random(seed).shuffle(permutation)
    return permutation


class GradientNoise2D:
    """Вычислить непрерывное поле шума на решётке с шагом 1 и периодом 256 по каждой оси."""

    def __init__(self, permutation):
        values = np.asarray(permutation)
        if (values.shape != (256,) or values.dtype.kind not in "iu"
                or not np.array_equal(np.sort(values), np.arange(256))):
            raise ValueError("Таблица шума должна содержать перестановку целых чисел 0–255.")
        self._permutation = np.tile(values.astype(np.int64), 2)

    def __call__(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("Координаты шума должны быть конечными числами.")
        floor_x, floor_y = np.floor(x), np.floor(y)
        ix = np.remainder(floor_x, 256).astype(np.int64)
        iy = np.remainder(floor_y, 256).astype(np.int64)
        dx, dy = x - floor_x, y - floor_y
        # По формуле Перлина (2002, раздел 3) первая и вторая производные равны нулю в точках 0 и 1.
        u = 6 * dx**5 - 15 * dx**4 + 10 * dx**3
        v = 6 * dy**5 - 15 * dy**4 + 10 * dy**3
        result = np.zeros(x.shape)
        for corner_x, weight_x in ((0, 1 - u), (1, u)):
            for corner_y, weight_y in ((0, 1 - v), (1, v)):
                hashed = self._permutation[self._permutation[ix + corner_x] + iy + corner_y]
                gradient = _GRADIENTS[hashed % len(_GRADIENTS)]
                projection = gradient[..., 0] * (dx - corner_x) + gradient[..., 1] * (dy - corner_y)
                result += weight_x * weight_y * projection
        return result


def normalized_octave_noise(x_m, y_m, noise, *, octaves=OCTAVES,
                            persistence=PERSISTENCE, lacunarity=LACUNARITY,
                            wavelength_m=BASE_WAVELENGTH_M):
    """Вычислить спектральную сумму по методу из раздела 10.6.3 PBRT.

    Функция принимает двумерное поле шума и целое число октав. Сумма нормируется по весам.
    Фильтрация по производным и дробная последняя октава из трёхмерной модели PBRT здесь
    не используются. По умолчанию частота удваивается, в реализации PBRT множитель равен 1,99.
    """
    if type(octaves) is not int or not 1 <= octaves <= 16:
        raise ValueError("Количество октав должно быть целым числом 1–16.")
    if not np.isfinite(persistence) or not 0 < persistence <= 1:
        raise ValueError("Вес следующей октавы должен быть в интервале (0, 1].")
    if not np.isfinite(lacunarity) or not 1 <= lacunarity <= 4:
        raise ValueError("Множитель частоты должен быть в интервале [1, 4].")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("Базовый масштаб должен быть положительным конечным числом.")
    x, y = np.broadcast_arrays(np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float))
    total = np.zeros(x.shape)
    weight_sum = 0.0
    for octave in range(octaves):
        frequency = lacunarity**octave / wavelength_m
        weight = persistence**octave
        total += weight * noise(x * frequency, y * frequency)
        weight_sum += weight
    return total / weight_sum


def terrain_heights(x_m, y_m, noise: GradientNoise2D):
    """Получить высоты искусственной DEM с помощью спектрального синтеза шума Перлина."""
    return HEIGHT_ORIGIN_M + HEIGHT_AMPLITUDE_M * normalized_octave_noise(x_m, y_m, noise)


def generator_metadata(seed: int, permutation: list[int]) -> dict:
    return {"method": "perlin_2d_octaves", "implementation_version": 2,
            "algorithm_reference": PERLIN_DOI, "seed": seed, "permutation": permutation,
            "algorithm_references": [
                {"url": PERLIN_DOI, "role": "Gradient noise and quintic interpolation"},
                {"url": PBRT_NOISE_URL, "role": "Spectral synthesis: weighted sum of noise octaves"}],
            "spectral_synthesis": "Fixed integer octaves, normalized; no derivative-based antialiasing",
            "base_wavelength_m": BASE_WAVELENGTH_M, "height_origin_m": HEIGHT_ORIGIN_M,
            "height_amplitude_m": HEIGHT_AMPLITUDE_M, "octaves": OCTAVES,
            "persistence": PERSISTENCE, "lacunarity": LACUNARITY,
            "gradient_set": "8 evenly spaced unit vectors in 2D",
            "fade": "6*t^5 - 15*t^4 + 10*t^3",
            "normalization": "divide octave sum by sum of weights",
            "artificial_nodata_pixel_row_col": [17, 18]}
