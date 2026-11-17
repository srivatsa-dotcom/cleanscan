"""Small geometry helpers shared by reconstruction and the synthetic scenes."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from . import physics


def lathe(profile, segments: int = 40) -> o3d.geometry.TriangleMesh:
    """Closed surface of revolution about +z from a (radius, z) profile, bottom to top."""
    prof = np.asarray(profile, float)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    V = [[0.0, 0.0, prof[0, 1]]]
    for r, z in prof:
        V += [[r * np.cos(a), r * np.sin(a), z] for a in ang]
    V.append([0.0, 0.0, prof[-1, 1]])
    n, top = len(prof), len(V) - 1
    F = []
    ring = lambda i, k: 1 + i * segments + (k % segments)  # noqa: E731
    for k in range(segments):
        F.append([0, ring(0, k + 1), ring(0, k)])
        F.append([top, ring(n - 1, k), ring(n - 1, k + 1)])
        for i in range(n - 1):
            a, b, c, d = ring(i, k), ring(i, k + 1), ring(i + 1, k), ring(i + 1, k + 1)
            F += [[a, b, d], [a, d, c]]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.array(V)), o3d.utility.Vector3iVector(np.array(F)))
    vol, _, _ = physics._solid_integrals(np.asarray(m.vertices), np.asarray(m.triangles))
    if vol < 0:
        m.triangles = o3d.utility.Vector3iVector(np.asarray(m.triangles)[:, [0, 2, 1]])
    return m
