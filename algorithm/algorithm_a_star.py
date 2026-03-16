"""
A* path planning kernel implementation.

Creator: Orange JYJ
License: Apache-2.0
"""
from __future__ import annotations

import heapq
import math
from typing import Callable


MapPoint3 = tuple[float, float, float]


def _clamp_0_100(v: float) -> float:
    return max(0.0, min(100.0, float(v)))


def plan_astar_path_map_xyz(
    start_xyz: MapPoint3,
    end_xyz: MapPoint3,
    occupancy,
    occ_nx: int | None,
    occ_nz: int | None,
    heights,
    map_to_index: Callable[[float, int, str], int],
) -> list[MapPoint3]:
    """Plan an A* path in map XYZ.

    - If occupancy exists: run true 3D A* in free voxels (x, z_height, y_plane).
    - Else: run a 2D planar fallback on heightfield and interpolate Z.
    """

    sx, sy, sz = start_xyz
    ex, ey, ez = end_xyz

    if occupancy is not None and occ_nx is not None and occ_nz is not None:
        try:
            occ = occupancy
            nx = int(occ.shape[0])
            ny_h = int(occ.shape[1])
            nz = int(occ.shape[2])
        except Exception:
            return []

        if nx < 2 or ny_h < 2 or nz < 2:
            return []

        sx_i = map_to_index(sx, nx, "round")
        sy_i = map_to_index(sy, nz, "round")
        sz_i = map_to_index(sz, ny_h, "round")
        ex_i = map_to_index(ex, nx, "round")
        ey_i = map_to_index(ey, nz, "round")
        ez_i = map_to_index(ez, ny_h, "round")

        s3 = _nearest_free_voxel(occ, (sx_i, sz_i, sy_i))
        g3 = _nearest_free_voxel(occ, (ex_i, ez_i, ey_i))
        if s3 is None or g3 is None:
            return []

        path3 = _astar_3d_free_voxels(occ, s3, g3)
        if not path3:
            return []

        waypoints3: list[MapPoint3] = []
        for ix, iy_h, iz in path3:
            mx = float(ix) / float(nx - 1) * 100.0
            my = float(iz) / float(nz - 1) * 100.0
            mz = float(iy_h) / float(ny_h - 1) * 100.0
            waypoints3.append((_clamp_0_100(mx), _clamp_0_100(my), _clamp_0_100(mz)))
        return waypoints3

    if heights is not None:
        rows, cols = heights.shape
        nx = int(cols)
        ny = int(rows)
        blocked = [[False for _ in range(ny)] for _ in range(nx)]
    else:
        return []

    if nx < 2 or ny < 2:
        return []

    s_ix = map_to_index(sx, nx, "round")
    s_iy = map_to_index(sy, ny, "round")
    e_ix = map_to_index(ex, nx, "round")
    e_iy = map_to_index(ey, ny, "round")

    blocked[s_ix][s_iy] = False
    blocked[e_ix][e_iy] = False

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        return math.hypot(dx, dy)

    start = (s_ix, s_iy)
    goal = (e_ix, e_iy)

    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0, start))

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    push_id = 1

    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    found = False
    while open_heap:
        _f, _id, cur = heapq.heappop(open_heap)
        if cur == goal:
            found = True
            break

        cur_g = g_score.get(cur)
        if cur_g is None:
            continue

        cx, cy = cur
        for dx, dy in neighbors:
            nx_i = cx + dx
            ny_i = cy + dy
            if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny:
                continue
            if blocked[nx_i][ny_i]:
                continue

            step = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
            cand_g = cur_g + step
            old_g = g_score.get((nx_i, ny_i))
            if old_g is not None and cand_g >= old_g:
                continue

            g_score[(nx_i, ny_i)] = cand_g
            came_from[(nx_i, ny_i)] = cur
            f = cand_g + heuristic((nx_i, ny_i), goal)
            heapq.heappush(open_heap, (f, push_id, (nx_i, ny_i)))
            push_id += 1

    if not found:
        return []

    path_idx: list[tuple[int, int]] = [goal]
    cur = goal
    while cur != start:
        cur = came_from[cur]
        path_idx.append(cur)
    path_idx.reverse()

    waypoints: list[MapPoint3] = []
    n = len(path_idx)
    for i, (ix, iy) in enumerate(path_idx):
        mx = float(ix) / float(nx - 1) * 100.0
        my = float(iy) / float(ny - 1) * 100.0
        t = 0.0 if n <= 1 else float(i) / float(n - 1)
        mz = (1.0 - t) * float(sz) + t * float(ez)
        waypoints.append((_clamp_0_100(mx), _clamp_0_100(my), _clamp_0_100(mz)))

    return waypoints


def _nearest_free_voxel(occ, start: tuple[int, int, int]) -> tuple[int, int, int] | None:
    """Find nearest free voxel to the requested voxel by expanding BFS shells."""

    nx = int(occ.shape[0])
    ny_h = int(occ.shape[1])
    nz = int(occ.shape[2])

    sx, sy, sz = start
    sx = max(0, min(nx - 1, int(sx)))
    sy = max(0, min(ny_h - 1, int(sy)))
    sz = max(0, min(nz - 1, int(sz)))

    def is_free(node: tuple[int, int, int]) -> bool:
        x, y, z = node
        try:
            return bool(occ[x, y, z] == 0)
        except Exception:
            return False

    s = (sx, sy, sz)
    if is_free(s):
        return s

    # First try same (x,z), searching upward then downward along height.
    for y in range(sy, ny_h):
        node = (sx, y, sz)
        if is_free(node):
            return node
    for y in range(sy - 1, -1, -1):
        node = (sx, y, sz)
        if is_free(node):
            return node

    # Then expand in full 3D shells.
    visited = {s}
    queue: list[tuple[int, int, int]] = [s]
    head = 0
    neighbors = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]

    max_scan = min(nx * ny_h * nz, 50000)
    scanned = 0
    while head < len(queue) and scanned < max_scan:
        cx, cy, cz = queue[head]
        head += 1
        scanned += 1
        for dx, dy, dz in neighbors:
            nx_i = cx + dx
            ny_i = cy + dy
            nz_i = cz + dz
            if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny_h or nz_i < 0 or nz_i >= nz:
                continue
            node = (nx_i, ny_i, nz_i)
            if node in visited:
                continue
            if is_free(node):
                return node
            visited.add(node)
            queue.append(node)

    return None


def _astar_3d_free_voxels(
    occ,
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    nx = int(occ.shape[0])
    ny_h = int(occ.shape[1])
    nz = int(occ.shape[2])

    def is_free(node: tuple[int, int, int]) -> bool:
        x, y, z = node
        try:
            return bool(occ[x, y, z] == 0)
        except Exception:
            return False

    def heuristic(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        dx = float(a[0] - b[0])
        dy = float(a[1] - b[1])
        dz = float(a[2] - b[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    if not is_free(start) or not is_free(goal):
        return []

    neighbors: list[tuple[int, int, int, float]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                cost = math.sqrt(float(dx * dx + dy * dy + dz * dz))
                neighbors.append((dx, dy, dz, cost))

    open_heap: list[tuple[float, int, tuple[int, int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0, start))

    g_score: dict[tuple[int, int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    push_id = 1

    # Safety cap to prevent worst-case explosions.
    max_expand = min(nx * ny_h * nz, 250000)
    expanded = 0

    while open_heap and expanded < max_expand:
        _f, _id, cur = heapq.heappop(open_heap)
        expanded += 1
        if cur == goal:
            break

        cur_g = g_score.get(cur)
        if cur_g is None:
            continue

        cx, cy, cz = cur
        for dx, dy, dz, step_cost in neighbors:
            nx_i = cx + dx
            ny_i = cy + dy
            nz_i = cz + dz
            if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny_h or nz_i < 0 or nz_i >= nz:
                continue
            nxt = (nx_i, ny_i, nz_i)
            if not is_free(nxt):
                continue

            cand_g = cur_g + step_cost
            old_g = g_score.get(nxt)
            if old_g is not None and cand_g >= old_g:
                continue

            g_score[nxt] = cand_g
            came_from[nxt] = cur
            f = cand_g + heuristic(nxt, goal)
            heapq.heappush(open_heap, (f, push_id, nxt))
            push_id += 1

    if goal not in came_from and goal != start:
        return []

    path: list[tuple[int, int, int]] = [goal]
    cur = goal
    while cur != start:
        parent = came_from.get(cur)
        if parent is None:
            return []
        cur = parent
        path.append(cur)
    path.reverse()
    return path
