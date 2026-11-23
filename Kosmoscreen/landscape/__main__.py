from __future__ import annotations

from dataclasses import replace
import argparse
import json
import sys
from pathlib import Path

from .project import Project, ProjectError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kosmoscreen — 3D-ландшафт из локальных геоданных")
    sub = parser.add_subparsers(dest="command", required=True)
    gui = sub.add_parser("gui", help="Окно параметров")
    gui.add_argument("project", nargs="?")
    for command, help_text in (("check", "Проверить входные данные"), ("build", "Построить модель и записать STL"),
                               ("experiment", "Сравнить сетки и адаптацию деревьев")):
        item = sub.add_parser(command, help=help_text)
        item.add_argument("project")
        if command in ("check", "build"):
            item.add_argument("--worldcover", nargs="?", const="", metavar="GEOTIFF",
                              help="Дополнить деревья по WorldCover: местный файл или автоматическая загрузка")
        if command == "build":
            item.add_argument("--preview", action="store_true", help="Только сцена, без объединения и STL")
    demo = sub.add_parser("demo", help="Создать искусственный тестовый набор")
    demo.add_argument("directory", nargs="?", help="Папка набора; по умолчанию зависит от метода")
    demo.add_argument("--method", choices=["perlin", "analytic", "analytic-legacy"], default="perlin",
                      help="Шум Перлина, поверхность Франке или прежняя плоскость с холмом")
    demo.add_argument("--seed", type=int, default=42, help="Seed генератора рельефа (по умолчанию 42)")
    view = sub.add_parser("view", help="Открыть сохранённую сцену")
    view.add_argument("scene")
    view.add_argument("--mode", choices=["terrain", "scene", "united", "edges", "contours"], default="scene")
    view.add_argument("--contour-interval", type=float, default=10.0, help="Шаг изолиний в метрах")
    view.add_argument("--top", action="store_true", help="Вид сверху")
    view.add_argument("--screenshot")
    inspect = sub.add_parser("inspect", help="Проверить бинарный STL")
    inspect.add_argument("stl")
    args = parser.parse_args(argv)
    try:
        if args.command == "gui":
            from .gui import launch
            launch(args.project)
        elif args.command == "demo":
            from .demo import create_demo
            directory = args.directory or {"perlin": "examples/perlin", "analytic": "examples/franke",
                                           "analytic-legacy": "examples/analytic-legacy"}[args.method]
            print(create_demo(directory, method=args.method, seed=args.seed))
        elif args.command == "view":
            from .viewer import show
            show(args.scene, args.mode, args.screenshot, contour_interval=args.contour_interval, top_view=args.top)
        elif args.command == "inspect":
            from .mesh import read_stl, require_valid
            from .union import as_manifold
            mesh = read_stl(args.stl)
            report = require_valid(mesh)
            report["manifold_status"] = str(as_manifold(mesh).status())
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "experiment":
            from .experiments import run
            run(Project.load(args.project), print)
        else:
            from .pipeline import build_model, save_result, save_preview, validate_data
            project = Project.load(args.project)
            if args.worldcover is not None:
                project.parameters = replace(project.parameters, worldcover=True, trees=True)
                if args.worldcover:
                    project.data["inputs"]["worldcover"] = str(Path(args.worldcover).resolve())
            if args.command == "check":
                print(json.dumps(validate_data(project), ensure_ascii=False, indent=2))
            else:
                scene, mesh, report = build_model(project, print, preview=args.preview)
                if args.preview:
                    path = save_preview(scene, report, project.output)
                    print(f"Сцена готова: {path}")
                    print(f"Объекты: {report['object_counts']}")
                    return 0
                save_result(scene, mesh, report, project.output)
                print(f"Готово: {project.output / 'model.stl'}")
                print("Геометрия и записанный STL проверены. Независимая проверка в слайсере — отдельный этап.")
        return 0
    except (ProjectError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
