"""
Alternative VPython-based 3D engine driver.

Creator: Orange JYJ
License: Apache-2.0
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerrainConfig:
    size_x: int = 100
    size_y: int = 100
    cell_size: float = 1.0
    height_scale: float = 12.0


def run_3d_scene(config: TerrainConfig = TerrainConfig()) -> None:
    """Open a minimal VPython 3D window and render a heightfield terrain.

    Note: VPython opens a browser-based WebGL canvas when run from a normal
    Python script. This function is intentionally minimal and meant to be the
    3D scene entry point.
    """

    # Import inside the function so importing this module doesn't immediately
    # launch any UI.
    import math

    import numpy as np
    from vpython import canvas, color, rate, triangle, vec, vertex

    if config.size_x < 2 or config.size_y < 2:
        raise ValueError("size_x/size_y must be >= 2")

    scene = canvas(title="3D View (VPython)", width=1000, height=700, background=color.black)

    # Generate a demo heightfield (replace with your map/height data later).
    x = np.linspace(0.0, 2.0 * math.pi, config.size_x, dtype=float)
    y = np.linspace(0.0, 2.0 * math.pi, config.size_y, dtype=float)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    heights = (np.sin(xx) * np.cos(yy)) * config.height_scale

    # Build vertex grid.
    half_x = (config.size_x - 1) * config.cell_size * 0.5
    half_y = (config.size_y - 1) * config.cell_size * 0.5

    verts: list[list[vertex]] = []
    for j in range(config.size_y):
        row: list[vertex] = []
        for i in range(config.size_x):
            px = i * config.cell_size - half_x
            py = j * config.cell_size - half_y
            pz = float(heights[j, i])

            # Simple colormap by height.
            t = (pz / (config.height_scale + 1e-9) + 1.0) * 0.5
            vcol = vec(0.1 + 0.2 * t, 0.4 + 0.5 * t, 0.1 + 0.1 * t)

            row.append(vertex(pos=vec(px, pz, py), color=vcol))
        verts.append(row)

    # Build triangles (two per grid cell).
    for j in range(config.size_y - 1):
        for i in range(config.size_x - 1):
            v00 = verts[j][i]
            v10 = verts[j][i + 1]
            v01 = verts[j + 1][i]
            v11 = verts[j + 1][i + 1]

            triangle(v0=v00, v1=v10, v2=v11)
            triangle(v0=v00, v1=v11, v2=v01)

        # Yield occasionally so the browser UI stays responsive while building.
        if j % 10 == 0:
            rate(60)

    scene.center = vec(0, 0, 0)
    scene.range = max(half_x, half_y) * 0.75

    # Keep scene alive.
    while True:
        rate(60)
