"""Utilities for building a single static mesh prim for LiDAR localization."""

from __future__ import annotations

import math

from .static_scene_geometry import get_static_cuboids, get_static_cylinders


def create_static_localization_mesh(mesh_path: str = "/World/LocalizationStaticMesh") -> dict:
    """Create a merged static mesh prim for ray-cast localization."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available for localization mesh creation.")

    if stage.GetPrimAtPath(mesh_path).IsValid():
        stage.RemovePrim(mesh_path)

    points: list[tuple[float, float, float]] = []
    face_vertex_counts: list[int] = []
    face_vertex_indices: list[int] = []

    def add_triangle(i0: int, i1: int, i2: int):
        face_vertex_counts.append(3)
        face_vertex_indices.extend([i0, i1, i2])

    def add_box(center: tuple[float, float, float], size: tuple[float, float, float]):
        cx, cy, cz = center
        sx, sy, sz = size
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
        base_index = len(points)
        vertices = [
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz),
            (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz),
            (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz),
            (cx - hx, cy + hy, cz + hz),
        ]
        points.extend(vertices)
        triangles = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1),
            (1, 5, 6), (1, 6, 2),
            (2, 6, 7), (2, 7, 3),
            (3, 7, 4), (3, 4, 0),
        ]
        for tri in triangles:
            add_triangle(base_index + tri[0], base_index + tri[1], base_index + tri[2])

    def add_cylinder(center: tuple[float, float, float], radius: float, height: float, segments: int = 24):
        cx, cy, cz = center
        half_h = height / 2.0
        top_center_index = len(points)
        points.append((cx, cy, cz + half_h))
        bottom_center_index = len(points)
        points.append((cx, cy, cz - half_h))
        top_ring = []
        bottom_ring = []
        for idx in range(segments):
            theta = 2.0 * math.pi * idx / segments
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)
            top_ring.append(len(points))
            points.append((x, y, cz + half_h))
            bottom_ring.append(len(points))
            points.append((x, y, cz - half_h))

        for idx in range(segments):
            nxt = (idx + 1) % segments
            add_triangle(top_center_index, top_ring[idx], top_ring[nxt])
            add_triangle(bottom_center_index, bottom_ring[nxt], bottom_ring[idx])
            add_triangle(top_ring[idx], bottom_ring[idx], bottom_ring[nxt])
            add_triangle(top_ring[idx], bottom_ring[nxt], top_ring[nxt])

    for cuboid in get_static_cuboids():
        add_box(cuboid["center"], cuboid["size"])
    for cylinder in get_static_cylinders():
        add_cylinder(cylinder["center"], cylinder["radius"], cylinder["height"])

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.Imageable(mesh).MakeInvisible()

    return {
        "mesh_path": mesh_path,
        "num_vertices": len(points),
        "num_triangles": len(face_vertex_counts),
    }
