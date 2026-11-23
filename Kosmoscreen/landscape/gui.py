from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import copy
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .project import Parameters, Project, ProjectError


class Application:
    def __init__(self, root, project_path=None):
        self.root = root
        self.project = None
        self.result = None
        self.viewers = {}
        self.busy = False
        self.loading = False
        self.messages = queue.Queue()
        self.entries = {}
        root.title("Космоскрин — 3D-ландшафт")
        root.geometry("1080x820")
        root.minsize(960, 740)
        root.configure(bg="#edf1f4")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#edf1f4")
        style.configure("TLabel", background="#edf1f4", foreground="#233b49", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 4))
        style.configure("TCheckbutton", background="#edf1f4", font=("Segoe UI", 10))
        style.configure("TLabelframe", background="#edf1f4")
        style.configure("TLabelframe.Label", background="#edf1f4", foreground="#233b49", font=("Segoe UI", 11, "bold"))
        style.configure("Accent.TButton", foreground="white", background="#236c75")
        banner = tk.Frame(root, bg="#183541", height=86)
        banner.pack(fill="x")
        tk.Label(banner, text="КОСМОСКРИН", font=("Segoe UI", 24, "bold"), fg="#e5f1ed", bg="#183541").pack(anchor="w", padx=24, pady=(8, 0))
        tk.Label(banner, text="КОСМОСНИМОК  /  РЕЛЬЕФ  /  ПЕЧАТНАЯ МОДЕЛЬ", font=("Segoe UI", 9), fg="#a9c6c7", bg="#183541").pack(anchor="w", padx=26, pady=(0, 8))
        body = ttk.Frame(root, padding=(18, 10))
        body.pack(fill="both", expand=True)
        project_bar = ttk.Frame(body)
        project_bar.pack(fill="x", pady=(0, 7))
        self.title = tk.StringVar(value="Откройте проект JSON")
        ttk.Label(project_bar, textvariable=self.title, font=("Segoe UI", 12, "bold")).pack(side="left", fill="x", expand=True)
        self.open_button = ttk.Button(project_bar, text="Открыть проект", command=self.open_project)
        self.open_button.pack(side="left", padx=4)
        self.save_button = ttk.Button(project_bar, text="Сохранить как…", command=self.save_project)
        self.save_button.pack(side="left", padx=4)
        files = ttk.LabelFrame(body, text=" 01 · Исходные данные ", padding=(12, 6))
        files.pack(fill="x")
        self.paths = {}
        for row, (key, label) in enumerate((("dem", "Высоты · GeoTIFF"), ("rgb", "RGB · GeoTIFF"), ("objects", "Объекты · GeoJSON"))):
            var = tk.StringVar()
            self.paths[key] = var
            ttk.Label(files, text=label, width=23).grid(row=row, column=0, sticky="w", pady=3)
            entry = ttk.Entry(files, textvariable=var)
            entry.grid(row=row, column=1, sticky="ew", padx=8)
            ttk.Button(files, text="Выбрать", command=lambda key=key: self.browse_input(key)).grid(row=row, column=2)
            var.trace_add("write", self.invalidate)
        files.columnconfigure(1, weight=1)
        params = ttk.LabelFrame(body, text=" 02 · Территория и параметры ", padding=(12, 6))
        params.pack(fill="x", pady=7)
        fields = [("lon", "Долгота центра", 42.730), ("lat", "Широта центра", 43.907),
                  ("width_m", "Ширина участка, м", 1000), ("height_m", "Высота участка, м", 1000),
                  ("long_side_mm", "Сторона макета, мм", 180), ("base_mm", "Основание, мм", 3),
                  ("exaggeration", "Усиление рельефа", 1), ("grid_nodes", "Узлы длинной стороны", 201),
                  ("max_buildings", "Лимит зданий (0–10 000)", 30), ("max_trees", "Лимит деревьев (0–10 000)", 30),
                  ("seed", "Вариант расстановки деревьев", 42), ("tree_spacing_m", "Шаг деревьев, м", 8)]
        for i, (key, label, default) in enumerate(fields):
            row, column = divmod(i, 3)
            ttk.Label(params, text=label).grid(row=row*2, column=column, sticky="w", padx=8)
            var = tk.StringVar(value=str(default))
            self.entries[key] = var
            ttk.Entry(params, textvariable=var, width=25).grid(row=row*2+1, column=column, sticky="ew", padx=8, pady=(1, 3))
            var.trace_add("write", self.invalidate)
        for i in range(3):
            params.columnconfigure(i, weight=1)
        checks = ttk.Frame(params)
        checks.grid(row=8, column=0, columnspan=3, sticky="w", padx=8, pady=(4, 0))
        self.flags = {}
        for key, label, value in (("smooth", "Сглаживание 3×3", False), ("buildings", "Здания", True),
                                  ("trees", "Деревья", True), ("adapt_trees", "Адаптация деревьев к печати", True)):
            var = tk.BooleanVar(value=value)
            self.flags[key] = var
            ttk.Checkbutton(checks, text=label, variable=var).pack(side="left", padx=(0, 16))
            var.trace_add("write", self.invalidate)
        optional = ttk.LabelFrame(body, text=" Дополнительный параметр · WorldCover ", padding=(12, 6))
        optional.pack(fill="x", pady=(0, 7))
        enabled = tk.BooleanVar(value=False)
        self.flags["worldcover"] = enabled
        ttk.Checkbutton(optional, text="Добавить деревья по WorldCover", variable=enabled).grid(row=0, column=0, sticky="w")
        enabled.trace_add("write", self.invalidate)
        cover_path = tk.StringVar()
        self.paths["worldcover"] = cover_path
        ttk.Label(optional, text="GeoTIFF (пусто — автозагрузка)").grid(row=0, column=1, padx=(16, 0))
        ttk.Entry(optional, textvariable=cover_path).grid(row=0, column=2, sticky="ew", padx=8)
        ttk.Button(optional, text="Выбрать", command=lambda: self.browse_input("worldcover")).grid(row=0, column=3)
        optional.columnconfigure(2, weight=1)
        cover_path.trace_add("write", self.invalidate)
        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(0, 7))
        self.check_button = ttk.Button(actions, text="Проверить данные", command=lambda: self.start("check"))
        self.check_button.pack(side="left", padx=(0, 7))
        self.build_button = ttk.Button(actions, text="Построить сцену", style="Accent.TButton", command=lambda: self.start("build"))
        self.build_button.pack(side="left", padx=7)
        self.view_button = ttk.Button(actions, text="Открыть просмотр", command=self.view, state="disabled")
        self.view_button.pack(side="left", padx=7)
        self.export_button = ttk.Button(actions, text="Экспортировать STL", command=lambda: self.start("export"), state="disabled")
        self.export_button.pack(side="left", padx=7)
        self.status = tk.StringVar(value="Готов к работе")
        ttk.Label(body, textvariable=self.status).pack(anchor="w", pady=(0, 4))
        self.progress = ttk.Progressbar(body, mode="indeterminate")
        self.progress.pack(fill="x")
        self.log = tk.Text(body, height=7, bg="#fcfdfd", fg="#29424e", font=("Consolas", 10), relief="flat", padx=12, pady=10)
        self.log.pack(fill="both", expand=True, pady=(8, 0))
        self.log.configure(state="disabled")
        self.root.after(100, self.poll)
        if project_path:
            self.open_project(project_path)
        else:
            default = Path(__file__).resolve().parent.parent/"examples/kislovodsk/project.json"
            if not default.is_file():
                default = Path(__file__).resolve().parent.parent/"examples/synthetic/project.json"
            if default.is_file():
                self.open_project(default)

    def append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text+"\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def invalidate(self, *_):
        if not self.loading:
            self.result = None
            self.view_button.configure(state="disabled")
            self.export_button.configure(state="disabled")
            self.status.set("Чтобы применить изменения, нажмите «Построить сцену».")

    def open_project(self, path=None):
        if self.busy:
            return
        path = path or filedialog.askopenfilename(filetypes=[("Проект JSON", "*.json")])
        if not path:
            return
        try:
            self.project = Project.load(path)
            self.loading = True
            self.title.set(self.project.data.get("title", self.project.path.name))
            for key, var in self.paths.items():
                var.set(self.project.data["inputs"].get(key, ""))
            values = asdict(self.project.parameters)
            area = self.project.data["area"]
            if "bounds" in area:
                from rasterio.warp import transform
                crs, b = self.project.coordinates()
                lon, lat = transform(crs, "EPSG:4326", [(b[0]+b[2])/2], [(b[1]+b[3])/2])
                values.update(lon=lon[0], lat=lat[0], width_m=b[2]-b[0], height_m=b[3]-b[1])
            else:
                values.update(lon=area["center"][0], lat=area["center"][1], width_m=area.get("width_m", 1000), height_m=area.get("height_m", 1000))
            for key, var in self.entries.items():
                var.set(str(values[key]))
            for key, var in self.flags.items():
                var.set(values[key])
            self.append(f"Проект: {self.project.path}\nДанные: {self.project.data.get('dataset_kind', 'происхождение не указано')}")
        except (ProjectError, ValueError) as exc:
            messagebox.showerror("Не удалось открыть проект", str(exc))
        finally:
            self.loading = False
            self.invalidate()

    def browse_input(self, key):
        if self.busy or self.project is None:
            return
        path = filedialog.askopenfilename(filetypes=[("GeoJSON", "*.geojson *.json")] if key == "objects" else [("GeoTIFF", "*.tif *.tiff")])
        if path:
            self.paths[key].set(Path(os.path.relpath(path, self.project.path.parent)).as_posix())
            if key == "worldcover":
                self.flags["worldcover"].set(True)

    def snapshot(self):
        if self.project is None:
            raise ProjectError("Сначала откройте проект JSON.")
        data = copy.deepcopy(self.project.data)
        values = asdict(self.project.parameters)
        integers = {"grid_nodes", "max_buildings", "max_trees", "seed"}
        for key in values.keys() & self.entries.keys():
            text = self.entries[key].get().replace(",", ".")
            values[key] = int(text) if key in integers else float(text)
        values.update({key: var.get() for key, var in self.flags.items()})
        data["parameters"] = values
        data["inputs"] = {key: var.get() for key, var in self.paths.items() if var.get()}
        entered = {key: float(self.entries[key].get().replace(",", ".")) for key in ("lon", "lat", "width_m", "height_m")}
        # Сохраняем явно заданные границы без округления, если пользователь не менял область.
        if "bounds" in data["area"]:
            from rasterio.warp import transform
            crs, b = self.project.coordinates()
            lon, lat = transform(crs, "EPSG:4326", [(b[0]+b[2])/2], [(b[1]+b[3])/2])
            unchanged = all(abs(a-bb) < 1e-8 for a, bb in zip(entered.values(), (lon[0], lat[0], b[2]-b[0], b[3]-b[1])))
        else:
            unchanged = False
        if not unchanged:
            data["area"] = {"center": [entered["lon"], entered["lat"]], "width_m": entered["width_m"], "height_m": entered["height_m"],
                            **({"crs": data["area"]["crs"]} if "crs" in data["area"] else {})}
        project = Project(self.project.path, data, Parameters(**values))
        project.validate()
        return project

    def save_project(self):
        if self.busy:
            return
        try:
            project = self.snapshot()
            path = filedialog.asksaveasfilename(defaultextension=".json", initialfile=project.path.name, filetypes=[("JSON", "*.json")])
            if path:
                project.save(path)
                self.open_project(path)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Параметры проекта", str(exc))

    def start(self, action):
        if self.busy:
            return
        try:
            project = self.snapshot()
        except (ValueError, OSError) as exc:
            messagebox.showerror("Проверьте параметры", str(exc))
            return
        if action == "export" and self.result is None:
            return
        self.busy = True
        self.progress.start(10)
        for button in (self.check_button, self.build_button, self.view_button, self.export_button, self.open_button, self.save_button):
            button.configure(state="disabled")
        self.status.set("Выполняется…")
        # Блокируем редактирование на время расчёта, чтобы настройки в окне соответствовали результату.
        def freeze(widget):
            for child in widget.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Checkbutton)):
                    child.configure(state="disabled")
                freeze(child)
        freeze(self.root)
        previous = self.result
        if action == "build":
            self.result = None
        def worker():
            try:
                from .pipeline import build_model, prepare_print, save_result, save_preview, validate_data
                if action == "check":
                    report = validate_data(project)
                    self.messages.put(("log", f"Данные проверены: {report['crs']}, сетка {report['grid_shape']}"))
                    for item in report["objects"]:
                        if item["status"] == "skipped":
                            self.messages.put(("log", f"Пропущен {item['id']}: {item['reason']}"))
                    nodes = report["suggested_grid_nodes"]
                    if nodes is not None:
                        pixel = f"{max(report['dem']['pixel_size_m']):.3g}".replace(".", ",")
                        step = f"{report['grid_step_m']:.3g}".replace(".", ",")
                        self.messages.put(("log", f"Пиксель карты высот ≈ {pixel} м. Узлы длинной стороны: "
                                                   f"ориентир {nodes}, текущий шаг {step} м."))
                    result = previous
                elif action == "build":
                    result = build_model(project, lambda text: self.messages.put(("log", text)), preview=True)
                    save_preview(result[0], result[2], project.output)
                else:
                    result = previous
                    if result[1] is None:
                        scene, _, report = result
                        mesh = prepare_print(scene, report, lambda text: self.messages.put(("log", text)))
                        result = scene, mesh, report
                    save_result(*result, project.output)
                    self.messages.put(("log", f"STL: {project.output/'model.stl'}"))
                self.messages.put(("done", (result, project, action)))
            except Exception as exc:
                self.messages.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "log":
                    self.append(value)
                    self.status.set(value)
                elif kind in ("done", "error"):
                    self.busy = False
                    self.progress.stop()
                    def thaw(widget):
                        for child in widget.winfo_children():
                            if isinstance(child, (ttk.Entry, ttk.Checkbutton)):
                                child.configure(state="normal")
                            thaw(child)
                    thaw(self.root)
                    for button in (self.check_button, self.build_button, self.open_button, self.save_button):
                        button.configure(state="normal")
                    if kind == "done":
                        self.result, self.project, action = value
                        text = {"check": "Проверка данных завершена", "build": "Сцена готова. Объединение для печати выполняется только при экспорте STL.",
                                "export": "STL сохранён и проверен. Проверка слайсером выполняется отдельно."}[action]
                        self.status.set(text)
                        self.append(text)
                        if action == "build":
                            counts = self.result[2]["object_counts"]
                            if "worldcover" in self.result[2]:
                                self.append(f"Деревья по WorldCover: {self.result[2]['worldcover']['trees_added']}.")
                            self.append(f"Добавлено зданий: {counts['buildings_added']}; деревьев: {counts['trees_added']}.")
                        if action in ("build", "export"):
                            for mode, process in tuple(self.viewers.items()):
                                if process.poll() is None:
                                    self.view(mode)
                                else:
                                    del self.viewers[mode]
                    else:
                        self.result = None
                        self.status.set("Ошибка. Подробности в журнале.")
                        self.append(value)
                        messagebox.showerror("Построение остановлено", value)
                    if self.result:
                        self.view_button.configure(state="normal")
                        self.export_button.configure(state="normal")
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def view(self, mode="scene"):
        if self.result is None or self.busy:
            return
        # Отдельный процесс позволяет Tk и VTK независимо обрабатывать события своих окон.
        folder = self.project.output / "preview" if self.result[1] is None else self.project.output
        try:
            process = subprocess.Popen([sys.executable, "-m", "landscape", "view", str(folder/"scene.npz"), "--mode", mode],
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as exc:
            self.append(f"Не удалось открыть просмотр. {exc}")
            return
        previous = self.viewers.get(mode)
        self.viewers[mode] = process
        if previous is not None and previous.poll() is None:
            try:
                previous.terminate()
            except ProcessLookupError:
                pass


def launch(project_path=None):
    # Если переносимый Python находится в WindowsApps, загрузчик Tcl иногда недоступен,
    # хотя Python может читать сами сценарии. Для этого случая setup.ps1 копирует
    # ресурсы Tcl/Tk из дистрибутива Python в локальное виртуальное окружение.
    local_tcl = Path(sys.prefix)/"tcl"
    if (local_tcl/"tcl8.6"/"init.tcl").is_file():
        os.environ["TCL_LIBRARY"] = str(local_tcl/"tcl8.6")
        os.environ["TK_LIBRARY"] = str(local_tcl/"tk8.6")
    root = tk.Tk()
    Application(root, project_path)
    root.mainloop()

