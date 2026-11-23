from __future__ import annotations

import numpy as np
import manifold3d as manifold

from .mesh import Mesh, require_valid
from .project import ProjectError


def as_manifold(mesh):
    require_valid(mesh)
    body = manifold.Manifold(manifold.Mesh64(np.asarray(mesh.vertices, dtype=np.float64),
                                           np.asarray(mesh.faces, dtype=np.uint64)))
    if body.status() != manifold.Error.NoError or body.is_empty():
        raise ProjectError(f"manifold3d отклонил объект {mesh.name}: {body.status()}")
    return body


def unite(meshes, progress=lambda message: None):
    bodies = []
    for mesh in meshes:
        require_valid(mesh)
        bodies.append(as_manifold(mesh))
    progress(f"Объединение {len(bodies)} замкнутых тел")
    result = manifold.Manifold.batch_boolean(bodies, manifold.OpType.Add)
    if result.status() != manifold.Error.NoError or result.is_empty():
        # При сбое указываем первую неудачную операцию, сохраняя исходную геометрию для диагностики.
        running = bodies[0]
        for mesh, body in zip(meshes[1:], bodies[1:]):
            running = running + body
            if running.status() != manifold.Error.NoError or running.is_empty():
                raise ProjectError(f"Ошибка объединения при добавлении {mesh.name}: {running.status()}")
        raise ProjectError(f"Ошибка группового объединения: {result.status()}")
    components = len(result.decompose())
    if components != 1:
        raise ProjectError(f"Объединение дало {components} отдельных тел; проверьте контакт объектов с основанием.")
    output = result.to_mesh64()
    mesh = Mesh(output.vert_properties[:, :3], output.tri_verts, "united")
    return mesh, {"status": str(result.status()), "components": components, "volume_mm3": result.volume(),
                  "method": "manifold3d geometric union; no auto-repair"}
