from __future__ import annotations

from pathlib import Path
import os
import tempfile
import math

import numpy as np

from .pipeline import load_scene
from .contours import contour_lines, terrain_heights, top_faces
from .project import ProjectError


def show(path, mode="scene", screenshot=None, *, contour_interval=10.0, top_view=False):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"kosmoscreen-matplotlib"))
    import pyvista as pv
    from matplotlib import get_data_path
    from vtkmodules.vtkCommonCore import VTK_FONT_FILE

    # Встроенные шрифты VTK могут не содержать кириллицу. Этот шрифт входит в комплект
    # Matplotlib на всех поддерживаемых платформах и не требует отдельной установки.
    font_file = str(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
    mode_names = {"terrain": "РЕЛЬЕФ", "scene": "СЦЕНА", "united": "ОБЪЕДИНЁННАЯ МОДЕЛЬ",
                  "edges": "РЁБРА ТРЕУГОЛЬНИКОВ", "contours": "ИЗОЛИНИИ"}

    scene, united = load_scene(path)
    if not math.isfinite(contour_interval) or contour_interval <= 0:
        raise ProjectError("Шаг изолиний должен быть положительным конечным числом.")
    try:
        heights = terrain_heights(scene, path)
    except ProjectError:
        if mode == "contours":
            raise
        heights = None
    contour_cache = {}
    plotter = pv.Plotter(window_size=(1280, 900), off_screen=bool(screenshot), title="Космоскрин — просмотр ландшафта")
    plotter.set_background("#10202b", top="#273e4a")
    state = {"mode": mode, "interval": contour_interval, "top": top_view, "labels": True}
    width_mm, height_mm = scene.dimensions
    footprint = f"{width_mm:g} × {height_mm:g} мм".replace(".", ",")

    def ui_text(text, **kwargs):
        return plotter.add_text(text, font_file=font_file, **kwargs)

    def polydata(mesh, faces=None):
        faces = mesh.faces if faces is None else faces
        return pv.PolyData(mesh.vertices, np.column_stack((np.full(len(faces), 3), faces)).ravel())

    # Группируем грани по материалу без булева объединения. Три буфера с теми же гранями
    # отрисовываются быстрее, чем тысячи отдельных объектов VTK.
    groups = {}
    for kind in ("building", "trunk", "crown"):
        vertices, faces, offset = [], [], 0
        for object_kind, mesh in scene.objects:
            if object_kind == kind:
                vertices.append(mesh.vertices)
                faces.append(mesh.faces + offset)
                offset += len(mesh.vertices)
        if vertices:
            triangles = np.concatenate(faces)
            groups[kind] = pv.PolyData(np.concatenate(vertices),
                np.column_stack((np.full(len(triangles), 3), triangles)).ravel())

    def draw_objects(edges=False):
        colors = {"building": "#e8ded0", "trunk": "#826348", "crown": "#55986a"}
        for kind, data in groups.items():
            plotter.add_mesh(data, color=colors[kind], smooth_shading=kind == "crown",
                             ambient=.3, show_edges=edges, edge_color="#31434b")

    def draw_contours(surface):
        if heights is None:
            plotter.add_mesh(surface, color="#d7c7ac")
            ui_text("Нет сведений о высотах.\nПостройте сцену заново.", position="upper_right", font_size=12)
            return
        visible = heights[scene.terrain.faces[top_faces(scene.terrain)].ravel()]
        low, high = float(visible.min()), float(visible.max())
        plotter.add_mesh(surface, scalars=heights, cmap="terrain", clim=(low, max(high, low+.01)),
                         ambient=.65, diffuse=.35,
                         scalar_bar_args={"title": "Высота, м", "vertical": True,
                             "position_x": .88, "position_y": .22, "height": .55, "width": .1,
                             "color": "#e4eee9", "fmt": "%.0f"})
        for prop in (plotter.scalar_bar.GetTitleTextProperty(), plotter.scalar_bar.GetLabelTextProperty()):
            prop.SetFontFamily(VTK_FONT_FILE)
            prop.SetFontFile(font_file)
        step = state["interval"]
        if step not in contour_cache:
            contour_cache[step] = contour_lines(scene.terrain, heights, step)
        labels, positions = [], []
        for contour in contour_cache[step]:
            points = contour.segments.reshape(-1, 3).copy()
            points[:, 2] += .025  # Приподнимаем только отображаемые линии, чтобы они не мерцали при совпадении с поверхностью.
            n = len(contour.segments)
            lines = np.column_stack((np.full(n, 2), np.arange(0, 2*n, 2), np.arange(1, 2*n, 2)))
            plotter.add_mesh(pv.PolyData(points, lines=lines.ravel()), color="#243632",
                             line_width=2, lighting=False)
            positions.append(points[len(points)//2])
            labels.append(f"{contour.elevation_m:g} м".replace(".", ","))
        if positions and state["labels"]:
            plotter.add_point_labels(np.array(positions), labels, show_points=False,
                                    font_size=13, text_color="#10231e", shape_color="#f5f3dd",
                                    shape_opacity=.85, margin=3, always_visible=True, font_file=font_file)
        ui_text(f"Шаг изолиний: {step:g} м".replace(".", ",") +
                         "\n6 / I — шаг: 5 / 10 / 20 / 50 м   8 / L — подписи\n7 / T — вид сверху / объёмный" +
                         ("\nВ диапазоне высот нет уровней\nс выбранным шагом." if not positions else ""),
                         position="upper_right", font_size=12, color="#e4eee9")

    def draw(selected, reset=False):
        camera = plotter.camera_position if not reset else None
        plotter.clear()
        # Вызов clear() удаляет и объекты, и источники света. Восстанавливаем освещение для каждого режима.
        plotter.enable_lightkit()
        unavailable = selected == "united" and united is None
        if unavailable:
            selected = "scene"
        state["mode"] = selected
        edges = selected == "edges"
        if selected in ("united", "edges") and united is not None:
            plotter.add_mesh(polydata(united), color="#d7c7ac", show_edges=edges, edge_color="#31434b",
                             ambient=.25, diffuse=.8, specular=.15)
        elif edges:
            plotter.add_mesh(polydata(scene.terrain), color="#d7c7ac", show_edges=True,
                             edge_color="#31434b", ambient=.25)
            draw_objects(edges=True)
        else:
            top = top_faces(scene.terrain)
            plotter.add_mesh(polydata(scene.terrain, scene.terrain.faces[~top]), color="#b8ac92", ambient=.3)
            surface = polydata(scene.terrain, scene.terrain.faces[top])
            if selected == "contours":
                try:
                    draw_contours(surface)
                except ProjectError as exc:
                    print(str(exc))
                    ui_text("Слишком малый шаг изолиний.\nНажмите 6, чтобы изменить шаг.", position="upper_right", font_size=12)
            else:
                surface.active_texture_coordinates = scene.terrain.vertices[:, :2] / np.array(scene.dimensions)
                plotter.add_mesh(surface, texture=pv.numpy_to_texture(scene.texture), ambient=.4, diffuse=.6)
            if selected == "scene":
                draw_objects()
        ui_text(f"КОСМОСКРИН\nРазмер макета: {footprint}",
                position="upper_left", font_size=16, color="#e4eee9", font="arial")
        ui_text(f"{mode_names[selected]}\n"
                         "1 Рельеф   2 Сцена   4 Рёбра   5 Изолинии\n"
                         "6 Шаг   7 Вид сверху   8 Подписи   R Сброс ракурса",
                         position="lower_left", font_size=11, color="#b8cbc9", font="arial")
        plotter.add_axes()
        if state["top"]:
            plotter.show_bounds(bounds=(0, width_mm, 0, height_mm, 0, 0), show_zaxis=False,
                                xtitle="X, mm", ytitle="Y, mm", n_xlabels=4, n_ylabels=4,
                                location="outer", color="#b8cbc9", font_size=12, fmt="%.4g", render=False)
        if unavailable:
            ui_text("Объединённая модель готовится при экспорте STL.\nСейчас показана сцена.",
                             position="upper_right", font_size=11, color="#e4eee9")
        if reset:
            if state["top"]:
                plotter.view_xy()
                plotter.enable_parallel_projection()
            else:
                plotter.disable_parallel_projection()
                plotter.view_isometric()
            plotter.reset_camera()
            plotter.camera.zoom(1.15)
        elif camera:
            plotter.camera_position = camera
        plotter.render()

    def change_interval():
        steps = [5.0, 10.0, 20.0, 50.0]
        current = state["interval"]
        state["interval"] = steps[(steps.index(current)+1) % len(steps)] if current in steps else 5.0
        draw("contours")

    def toggle_top():
        state["top"] = not state["top"]
        draw(state["mode"], True)

    def reset():
        state["top"] = False
        draw(state["mode"], True)

    def toggle_labels():
        state["labels"] = not state["labels"]
        draw(state["mode"])

    for key, selected in zip("1245", ("terrain", "scene", "edges", "contours")):
        plotter.add_key_event(key, lambda selected=selected: draw(selected))
    plotter.add_key_event("r", reset)
    plotter.add_key_event("i", change_interval)
    plotter.add_key_event("6", change_interval)
    plotter.add_key_event("t", toggle_top)
    plotter.add_key_event("7", toggle_top)
    plotter.add_key_event("l", toggle_labels)
    plotter.add_key_event("8", toggle_labels)
    draw(mode, True)
    if screenshot:
        Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(screenshot), auto_close=True)
    else:
        plotter.show()

