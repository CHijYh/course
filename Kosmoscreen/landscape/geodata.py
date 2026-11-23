from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import rasterio
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window

from .project import Project, ProjectError


def bilinear(values, cols, rows):
    """Интерполировать по центрам пикселей. Для многоканального растра оси массива — строки, столбцы и каналы."""
    cols, rows = np.asarray(cols, dtype=float), np.asarray(rows, dtype=float)
    if (np.any(cols < -1e-7) or np.any(rows < -1e-7)
            or np.any(cols > values.shape[1]-1+1e-7) or np.any(rows > values.shape[0]-1+1e-7)):
        raise ProjectError("Растр не покрывает участок с запасом для интерполяции (нужен минимум один пиксель).")
    cols = np.clip(cols, 0, values.shape[1]-1)
    rows = np.clip(rows, 0, values.shape[0]-1)
    c = np.minimum(np.floor(cols).astype(int), values.shape[1]-2)
    r = np.minimum(np.floor(rows).astype(int), values.shape[0]-2)
    u, v = cols-c, rows-r
    if values.ndim == 3:
        u, v = u[..., None], v[..., None]
    return ((1-u)*(1-v)*values[r, c] + u*(1-v)*values[r, c+1]
            + (1-u)*v*values[r+1, c] + u*v*values[r+1, c+1])


def fill_nodata(values, invalid, inside, affine=None, metric_centers=None):
    """Заполнить пропуски ближайшими исходными измерениями. При равных расстояниях выбрать первое по порядку обхода."""
    inside_count = int(np.count_nonzero(inside))
    missing = int(np.count_nonzero(invalid & inside))
    if not inside_count:
        raise ProjectError("Внутри участка нет центров пикселей DEM: выберите более крупный участок.")
    fraction = missing / inside_count
    if fraction > 0.05 + 1e-12:
        raise ProjectError(f"Пропуски DEM внутри участка: {fraction:.1%}; допускается не более 5%.")
    result = values.copy()
    good = np.argwhere(~invalid)
    bad = np.argwhere(invalid)
    if not len(good):
        raise ProjectError("В DEM нет корректных высот.")
    # На небольшом фрагменте DEM можно сравнить расстояния до всех центров с исходными измерениями.
    def positions(indices):
        if metric_centers is not None:
            return metric_centers[indices[:, 0], indices[:, 1]]
        if affine is None:
            return indices.astype(float)
        r, c = indices[:, 0], indices[:, 1]
        return np.column_stack((affine.a*c + affine.b*r, affine.d*c + affine.e*r))
    gp = positions(good)
    for rc, position in zip(bad, positions(bad)):
        distance2 = np.sum((gp-position)**2, axis=1)
        donor = good[int(np.argmin(distance2))]
        result[tuple(rc)] = values[tuple(donor)]
    return result, {"missing_inside": missing, "pixels_inside": inside_count,
                    "missing_fraction": fraction, "filled_halo": len(bad)-missing,
                    "fill_method": "nearest_original_valid_pixel"}


def smooth_once(values):
    result = values.copy()
    result[1:-1, 1:-1] = sum(values[j:values.shape[0]-2+j, i:values.shape[1]-2+i]
                             for j in range(3) for i in range(3)) / 9
    return result


@dataclass
class Raster:
    values: np.ndarray
    affine: object
    crs: object
    metadata: dict

    def sample(self, x, y, working_crs):
        shape = np.asarray(x).shape
        sx, sy = transform(working_crs, self.crs, np.asarray(x).ravel().tolist(), np.asarray(y).ravel().tolist())
        inverse = ~self.affine
        sx, sy = np.asarray(sx), np.asarray(sy)
        c = inverse.a*sx + inverse.b*sy + inverse.c
        r = inverse.d*sx + inverse.e*sy + inverse.f
        value = bilinear(self.values, c-0.5, r-0.5)
        return value.reshape(shape + (() if self.values.ndim == 2 else (self.values.shape[2],)))


def load_raster(path, working_crs, bounds, *, rgb=False):
    try:
        with rasterio.open(path) as source:
            if source.crs is None:
                raise ProjectError(f"У файла {path.name} отсутствует CRS.")
            if source.count < (3 if rgb else 1):
                raise ProjectError(f"Файл {path.name}: недостаточно каналов.")
            if source.width < 2 or source.height < 2:
                raise ProjectError("Размер растра должен быть не менее 2 × 2 пикселей.")
            if abs(source.transform.determinant) < 1e-20:
                raise ProjectError("Вырожденная геопривязка растра.")
            # Дополнительные точки вдоль границ нужны также для повёрнутых растров и географических систем координат.
            left, bottom, right, top = transform_bounds(working_crs, source.crs, *bounds, densify_pts=41)
            inv = ~source.transform
            corners = [(inv.a*x+inv.b*y+inv.c, inv.d*x+inv.e*y+inv.f) for x in (left, right) for y in (bottom, top)]
            c0, r0 = (max(0, math.floor(min(p[k] for p in corners))-2) for k in (0, 1))
            c1 = min(source.width, math.ceil(max(p[0] for p in corners))+2)
            r1 = min(source.height, math.ceil(max(p[1] for p in corners))+2)
            if c1-c0 < 2 or r1-r0 < 2:
                raise ProjectError(f"{path.name} не пересекается с участком.")
            if (c1-c0)*(r1-r0) > 4_000_000:
                raise ProjectError("Обрезка растра превышает 4 млн пикселей. Подготовьте менее подробный GeoTIFF.")
            window = Window(c0, r0, c1-c0, r1-r0)
            affine = source.window_transform(window)
            indexes = [1, 2, 3] if rgb else [1]
            data = source.read(indexes, window=window, masked=True)
            # Ноль в одном из каналов RGB допустим даже у пикселя с данными.
            # Используем маску набора данных: альфа-канал, внутреннюю маску или объединение масок валидности RGB по ИЛИ.
            invalid = ((source.dataset_mask(window=window) == 0) if rgb else np.any(np.ma.getmaskarray(data), axis=0))
            invalid = invalid | ~np.all(np.isfinite(data.data), axis=0)
            values = data.data.astype(float)
            for i, band in enumerate(indexes):
                values[i] = values[i]*source.scales[band-1]+source.offsets[band-1]
            metadata = {"file": path.name, "crs": str(source.crs), "resolution": list(source.res),
                        "crop_shape": list(invalid.shape), "nodata": str(source.nodata)}
            if rgb:
                if invalid.any():
                    # Пропуски в области выборки проверяются по отдельной маске.
                    values[:, invalid] = 0
                metadata["source_dtype"] = source.dtypes[0]
                raster = Raster(np.moveaxis(values, 0, -1), affine, source.crs, metadata)
                raster.mask = Raster(invalid.astype(float), affine, source.crs, {})
                return raster
            rows, cols = np.indices(invalid.shape)
            x = affine.a*(cols+.5) + affine.b*(rows+.5) + affine.c
            y = affine.d*(cols+.5) + affine.e*(rows+.5) + affine.f
            wx, wy = transform(source.crs, working_crs, x.ravel().tolist(), y.ravel().tolist())
            wx, wy = np.asarray(wx).reshape(x.shape), np.asarray(wy).reshape(y.shape)
            inside = (wx >= bounds[0]) & (wx <= bounds[2]) & (wy >= bounds[1]) & (wy <= bounds[3])
            centers_m = np.stack((wx, wy), axis=-1)
            filled, stats = fill_nodata(values[0], invalid, inside, affine, centers_m)
            row, col = ((length - 1) // 2 for length in invalid.shape)
            center = centers_m[row, col]
            metadata["pixel_size_m"] = [math.dist(center, centers_m[row, col + 1]),
                                        math.dist(center, centers_m[row + 1, col])]
            metadata.update(stats)
            return Raster(filled, affine, source.crs, metadata)
    except rasterio.errors.RasterioError as exc:
        raise ProjectError(f"Ошибка чтения GeoTIFF {path}: {exc}") from exc


def load_inputs(project: Project):
    project.validate()
    crs, bounds = project.coordinates()
    dem = load_raster(project.file("dem"), crs, bounds)
    rgb = load_raster(project.file("rgb"), crs, bounds, rgb=True)
    return crs, bounds, dem, rgb


def texture_image(rgb, crs, bounds, long_side=512):
    w, h = bounds[2]-bounds[0], bounds[3]-bounds[1]
    nx, ny = max(2, round(long_side*w/max(w, h))), max(2, round(long_side*h/max(w, h)))
    xx, yy = np.meshgrid(np.linspace(bounds[0], bounds[2], nx), np.linspace(bounds[3], bounds[1], ny))
    if np.any(rgb.mask.sample(xx, yy, crs) > 1e-10):
        raise ProjectError("RGB-снимок содержит пропуски внутри участка. Подготовьте полный снимок.")
    values = rgb.sample(xx, yy, crs)
    if rgb.metadata["source_dtype"] != "uint8":
        low, high = np.percentile(values, [2, 98])
        if high > low:
            values = 255*(values-low)/(high-low)
        rgb.metadata["display_stretch"] = {"percentiles": [2, 98], "range": [float(low), float(high)]}
    return np.clip(values, 0, 255).astype(np.uint8)

