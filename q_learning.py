from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import Callable


def _clamp_0_100(v: float) -> float:
    return max(0.0, min(100.0, float(v)))


@dataclass
class QLearningModel:
    q_table: object
    policy: object
    free_mask: object
    goal_idx: tuple[int, ...]
    grid_shape: tuple[int, ...]
    is_3d: bool
    actions: tuple[tuple[int, ...], ...]
    distance_map: object


def _actions_3d_26() -> tuple[tuple[int, int, int], ...]:
    acts = []
    for dx in (-1, 0, 1):
        for dy_h in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy_h == 0 and dz == 0:
                    continue
                acts.append((dx, dy_h, dz))
    return tuple(acts)


def _move_cost_3d(delta: tuple[int, int, int]) -> float:
    dx, dy_h, dz = delta
    # Euclidean unit-grid move cost: axis=1, planar diagonal=sqrt(2), space diagonal=sqrt(3)
    return float(math.sqrt(float(dx * dx + dy_h * dy_h + dz * dz)))


def _is_free_3d(free_mask, x: int, y_h: int, z: int, nx: int, ny_h: int, nz: int) -> bool:
    return 0 <= x < nx and 0 <= y_h < ny_h and 0 <= z < nz and bool(free_mask[x, y_h, z])


def _valid_transition_3d(
    free_mask,
    x: int,
    y_h: int,
    z: int,
    dx: int,
    dy_h: int,
    dz: int,
    nx: int,
    ny_h: int,
    nz: int,
) -> bool:
    tx = x + dx
    ty = y_h + dy_h
    tz = z + dz
    if not _is_free_3d(free_mask, tx, ty, tz, nx, ny_h, nz):
        return False

    # Anti corner-cutting: when taking diagonal components, require each axis-adjacent
    # intermediate voxel to be free as well.
    if dx != 0 and not _is_free_3d(free_mask, x + dx, y_h, z, nx, ny_h, nz):
        return False
    if dy_h != 0 and not _is_free_3d(free_mask, x, y_h + dy_h, z, nx, ny_h, nz):
        return False
    if dz != 0 and not _is_free_3d(free_mask, x, y_h, z + dz, nx, ny_h, nz):
        return False
    return True


def nearest_free_cell_2d(free_mask, start: tuple[int, int]) -> tuple[int, int] | None:
    nx = int(free_mask.shape[0])
    ny = int(free_mask.shape[1])
    sx = max(0, min(nx - 1, int(start[0])))
    sy = max(0, min(ny - 1, int(start[1])))

    if bool(free_mask[sx, sy]):
        return (sx, sy)

    queue = [(sx, sy)]
    visited = {(sx, sy)}
    head = 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while head < len(queue):
        cx, cy = queue[head]
        head += 1
        for dx, dy in neighbors:
            nx_i = cx + dx
            ny_i = cy + dy
            if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny:
                continue
            node = (nx_i, ny_i)
            if node in visited:
                continue
            if bool(free_mask[nx_i, ny_i]):
                return node
            visited.add(node)
            queue.append(node)
    return None


def nearest_free_voxel_3d(
    free_mask,
    start: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    nx = int(free_mask.shape[0])
    ny_h = int(free_mask.shape[1])
    nz = int(free_mask.shape[2])

    sx = max(0, min(nx - 1, int(start[0])))
    sy_h = max(0, min(ny_h - 1, int(start[1])))
    sz = max(0, min(nz - 1, int(start[2])))

    if bool(free_mask[sx, sy_h, sz]):
        return (sx, sy_h, sz)

    # Prefer staying on the same planar location and searching height first.
    for y_h in range(sy_h, ny_h):
        if bool(free_mask[sx, y_h, sz]):
            return (sx, y_h, sz)
    for y_h in range(sy_h - 1, -1, -1):
        if bool(free_mask[sx, y_h, sz]):
            return (sx, y_h, sz)

    queue = [(sx, sy_h, sz)]
    visited = {(sx, sy_h, sz)}
    head = 0
    neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    while head < len(queue):
        cx, cy_h, cz = queue[head]
        head += 1
        for dx, dy_h, dz in neighbors:
            nx_i = cx + dx
            ny_i = cy_h + dy_h
            nz_i = cz + dz
            if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny_h or nz_i < 0 or nz_i >= nz:
                continue
            node = (nx_i, ny_i, nz_i)
            if node in visited:
                continue
            if bool(free_mask[nx_i, ny_i, nz_i]):
                return node
            visited.add(node)
            queue.append(node)

    return None


def _bfs_distance_2d(free_mask, goal: tuple[int, int], progress_callback: Callable[[int], None] | None = None):
    import numpy as np

    nx, ny = int(free_mask.shape[0]), int(free_mask.shape[1])
    dist = np.full((nx, ny), np.inf, dtype=np.float32)
    gx, gy = goal

    if not bool(free_mask[gx, gy]):
        return dist

    dist[gx, gy] = 0.0
    queue = [(gx, gy)]
    head = 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    total = max(1, int(np.count_nonzero(free_mask)))

    while head < len(queue):
        cx, cy = queue[head]
        head += 1
        base = float(dist[cx, cy])
        for dx, dy in neighbors:
            nx_i = cx + dx
            ny_i = cy + dy
            if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny:
                continue
            if not bool(free_mask[nx_i, ny_i]):
                continue
            if dist[nx_i, ny_i] <= base + 1.0:
                continue
            dist[nx_i, ny_i] = base + 1.0
            queue.append((nx_i, ny_i))

        if progress_callback is not None and head % 256 == 0:
            try:
                progress_callback(int(60 * min(1.0, head / total)))
            except Exception:
                pass

    return dist


def _bfs_distance_3d(free_mask, goal: tuple[int, int, int], progress_callback: Callable[[int], None] | None = None):
    import numpy as np

    nx, ny_h, nz = int(free_mask.shape[0]), int(free_mask.shape[1]), int(free_mask.shape[2])
    dist = np.full((nx, ny_h, nz), np.inf, dtype=np.float32)
    gx, gy_h, gz = goal

    if not bool(free_mask[gx, gy_h, gz]):
        return dist

    actions = _actions_3d_26()
    costs = [_move_cost_3d(a) for a in actions]

    dist[gx, gy_h, gz] = 0.0
    heap = [(0.0, gx, gy_h, gz)]
    popped = 0
    total = max(1, int(np.count_nonzero(free_mask)))

    while heap:
        base, cx, cy_h, cz = heapq.heappop(heap)
        if base > float(dist[cx, cy_h, cz]) + 1e-9:
            continue
        popped += 1

        for (dx, dy_h, dz), step_cost in zip(actions, costs):
            nx_i = cx + dx
            ny_i = cy_h + dy_h
            nz_i = cz + dz
            if not _valid_transition_3d(free_mask, cx, cy_h, cz, dx, dy_h, dz, nx, ny_h, nz):
                continue
            cand = float(base) + float(step_cost)
            if dist[nx_i, ny_i, nz_i] <= cand:
                continue
            dist[nx_i, ny_i, nz_i] = cand
            heapq.heappush(heap, (cand, nx_i, ny_i, nz_i))

        if progress_callback is not None and popped % 2048 == 0:
            try:
                progress_callback(int(60 * min(1.0, popped / total)))
            except Exception:
                pass

    return dist


def _value_from_distance(dist, gamma: float, goal_reward: float, step_penalty: float):
    import numpy as np

    values = np.full(dist.shape, -1e9, dtype=np.float32)
    finite = np.isfinite(dist)
    if not finite.any():
        return values

    d = dist[finite].astype(np.float32)
    # Discounted return with terminal reward and per-step penalty.
    # V(d) = gamma^d * goal_reward + step_penalty * sum_{k=0}^{d-1} gamma^k
    gpow = np.power(gamma, d, dtype=np.float32)
    denom = max(1e-9, 1.0 - gamma)
    step_sum = (1.0 - gpow) / denom
    v = gpow * float(goal_reward) + float(step_penalty) * step_sum
    values[finite] = v.astype(np.float32)
    return values


def _build_q_from_values_2d(free_mask, values, goal: tuple[int, int], gamma: float):
    import numpy as np

    nx, ny = int(free_mask.shape[0]), int(free_mask.shape[1])
    actions = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    q = np.full((nx, ny, len(actions)), -1e9, dtype=np.float32)
    gx, gy = goal

    for sx, sy in np.argwhere(free_mask):
        sx = int(sx)
        sy = int(sy)
        for a, (dx, dy) in enumerate(actions):
            nx_i = sx + dx
            ny_i = sy + dy
            reward = -1.0
            if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny or not bool(free_mask[nx_i, ny_i]):
                nx_i, ny_i = sx, sy
                reward = -8.0
            elif (nx_i, ny_i) == (gx, gy):
                reward = 200.0
            q[sx, sy, a] = float(reward) + float(gamma) * float(values[nx_i, ny_i])

    return q, actions


def _build_q_from_values_3d(free_mask, values, goal: tuple[int, int, int], gamma: float):
    import numpy as np

    nx, ny_h, nz = int(free_mask.shape[0]), int(free_mask.shape[1]), int(free_mask.shape[2])
    actions = _actions_3d_26()
    costs = [_move_cost_3d(a) for a in actions]
    q = np.full((nx, ny_h, nz, len(actions)), -1e9, dtype=np.float32)
    gx, gy_h, gz = goal

    for sx, sy_h, sz in np.argwhere(free_mask):
        sx = int(sx)
        sy_h = int(sy_h)
        sz = int(sz)
        for a, ((dx, dy_h, dz), step_cost) in enumerate(zip(actions, costs)):
            nx_i = sx + dx
            ny_i = sy_h + dy_h
            nz_i = sz + dz
            reward = -float(step_cost)
            if not _valid_transition_3d(free_mask, sx, sy_h, sz, dx, dy_h, dz, nx, ny_h, nz):
                nx_i, ny_i, nz_i = sx, sy_h, sz
                reward = -8.0
            elif (nx_i, ny_i, nz_i) == (gx, gy_h, gz):
                reward = 260.0
            q[sx, sy_h, sz, a] = float(reward) + float(gamma) * float(values[nx_i, ny_i, nz_i])

    return q, actions


def train_q_learning_full_map(
    occupancy,
    heights,
    end_map_xyz: tuple[float, float, float] | None,
    map_to_index: Callable[[float, int, str], int],
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[bool, QLearningModel | None, int, int, tuple[int, int]]:
    import numpy as np

    if end_map_xyz is None:
        return (False, None, 0, 0, (0, 0))

    if progress_callback is not None:
        try:
            progress_callback(0)
        except Exception:
            pass

    gamma = 0.95
    step_penalty = -1.0

    if occupancy is not None:
        occ = np.asarray(occupancy)
        if occ.ndim != 3:
            return (False, None, 0, 0, (0, 0))
        nx, ny_h, nz = int(occ.shape[0]), int(occ.shape[1]), int(occ.shape[2])
        free_mask = np.asarray(occ == 0, dtype=bool)

        ex, ey, ez = end_map_xyz
        gx = map_to_index(ex, nx, "round")
        gy_plane = map_to_index(ey, nz, "round")
        gz_h = map_to_index(ez, ny_h, "round")
        goal3 = nearest_free_voxel_3d(free_mask, (gx, gz_h, gy_plane))
        if goal3 is None:
            return (False, None, 0, 0, (0, 0))

        free_cells = int(np.count_nonzero(free_mask))
        if free_cells <= 1:
            return (False, None, 0, 0, (0, 0))

        gx3, gy_h3, gz3 = goal3

        dist = _bfs_distance_3d(free_mask, goal3, progress_callback=progress_callback)
        values = _value_from_distance(dist, gamma=gamma, goal_reward=260.0, step_penalty=step_penalty)
        q, actions_3d = _build_q_from_values_3d(free_mask, values, goal3, gamma=gamma)
        episodes = int(np.max(dist[np.isfinite(dist)])) if np.isfinite(dist).any() else 0

        if progress_callback is not None:
            try:
                progress_callback(90)
            except Exception:
                pass

        model = QLearningModel(
            q_table=q,
            policy=np.argmax(q, axis=3).astype(np.int8),
            free_mask=free_mask,
            goal_idx=(gx3, gy_h3, gz3),
            grid_shape=(nx, ny_h, nz),
            is_3d=True,
            actions=actions_3d,
            distance_map=dist,
        )

        if progress_callback is not None:
            try:
                progress_callback(100)
            except Exception:
                pass

        # Keep 2-value summary for existing UI text: (x_idx, y_plane_idx)
        return (True, model, episodes, free_cells, (gx3, gz3))
    elif heights is not None:
        rows, cols = heights.shape
        nx, nz = int(cols), int(rows)
        free_mask = np.ones((nx, nz), dtype=bool)
    else:
        return (False, None, 0, 0, (0, 0))

    if nx < 2 or nz < 2:
        return (False, None, 0, 0, (0, 0))

    ex, ey, _ = end_map_xyz
    gx = map_to_index(ex, nx, "round")
    gy = map_to_index(ey, nz, "round")
    goal = nearest_free_cell_2d(free_mask, (gx, gy))
    if goal is None:
        return (False, None, 0, 0, (0, 0))

    gx, gy = goal
    free_cells = int(np.count_nonzero(free_mask))
    if free_cells <= 1:
        return (False, None, 0, 0, (0, 0))

    dist = _bfs_distance_2d(free_mask, (gx, gy), progress_callback=progress_callback)
    values = _value_from_distance(dist, gamma=gamma, goal_reward=200.0, step_penalty=step_penalty)
    q, actions_2d = _build_q_from_values_2d(free_mask, values, (gx, gy), gamma=gamma)
    episodes = int(np.max(dist[np.isfinite(dist)])) if np.isfinite(dist).any() else 0

    if progress_callback is not None:
        try:
            progress_callback(90)
        except Exception:
            pass

    model = QLearningModel(
        q_table=q,
        policy=np.argmax(q, axis=2).astype(np.int8),
        free_mask=free_mask,
        goal_idx=(gx, gy),
        grid_shape=(nx, nz),
        is_3d=False,
        actions=actions_2d,
        distance_map=dist,
    )
    if progress_callback is not None:
        try:
            progress_callback(100)
        except Exception:
            pass
    return (True, model, episodes, free_cells, (gx, gy))


def build_q_learning_path(
    model: QLearningModel,
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    map_to_index: Callable[[float, int, str], int],
) -> list[tuple[float, float, float]]:
    import numpy as np

    if model.is_3d:
        nx, ny_h, nz = model.grid_shape
        if nx < 2 or ny_h < 2 or nz < 2:
            return []

        sx = map_to_index(start_xyz[0], nx, "round")
        sy_plane = map_to_index(start_xyz[1], nz, "round")
        sz_h = map_to_index(start_xyz[2], ny_h, "round")
        start = nearest_free_voxel_3d(model.free_mask, (sx, sz_h, sy_plane))
        if start is None:
            return []

        if not np.isfinite(model.distance_map[start[0], start[1], start[2]]):
            return []

        if len(model.goal_idx) != 3:
            return []
        gx, gy_h, gz = model.goal_idx
        actions_3d = model.actions

        path_idx = [start]
        visited = {start}
        cx, cy_h, cz = start
        max_steps = max(1000, min(12000, nx * ny_h * nz // 2))

        for _ in range(max_steps):
            if (cx, cy_h, cz) == (gx, gy_h, gz):
                break

            a = int(model.policy[cx, cy_h, cz])
            if a < 0 or a >= len(actions_3d):
                return []

            dx, dy_h, dz = actions_3d[a]
            nx_i = cx + dx
            ny_i = cy_h + dy_h
            nz_i = cz + dz
            if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny_h or nz_i < 0 or nz_i >= nz:
                return []
            if not bool(model.free_mask[nx_i, ny_i, nz_i]):
                return []

            cur_d = float(model.distance_map[cx, cy_h, cz])
            next_d = float(model.distance_map[nx_i, ny_i, nz_i])
            if not math.isfinite(next_d) or next_d >= cur_d:
                # Fallback: pick a globally descending neighbor by distance.
                best = None
                best_d = float("inf")
                for bdx, bdy_h, bdz in actions_3d:
                    bx = cx + bdx
                    by = cy_h + bdy_h
                    bz = cz + bdz
                    if bx < 0 or bx >= nx or by < 0 or by >= ny_h or bz < 0 or bz >= nz:
                        continue
                    if not bool(model.free_mask[bx, by, bz]):
                        continue
                    bd = float(model.distance_map[bx, by, bz])
                    if not math.isfinite(bd):
                        continue
                    if bd < best_d:
                        best_d = bd
                        best = (bx, by, bz)
                if best is None or best_d >= cur_d:
                    return []
                nx_i, ny_i, nz_i = best

            node = (nx_i, ny_i, nz_i)
            if node in visited:
                return []
            visited.add(node)
            path_idx.append(node)
            cx, cy_h, cz = node

        if not path_idx or path_idx[-1] != (gx, gy_h, gz):
            return []

        planned: list[tuple[float, float, float]] = []
        for ix, iy_h, iz in path_idx:
            mx = float(ix) / float(nx - 1) * 100.0
            my = float(iz) / float(nz - 1) * 100.0
            mz = float(iy_h) / float(ny_h - 1) * 100.0
            planned.append((_clamp_0_100(mx), _clamp_0_100(my), _clamp_0_100(mz)))
        return planned

    nx, nz = model.grid_shape
    if nx < 2 or nz < 2:
        return []

    sx = map_to_index(start_xyz[0], nx, "round")
    sy = map_to_index(start_xyz[1], nz, "round")
    start = nearest_free_cell_2d(model.free_mask, (sx, sy))
    if start is None:
        return []

    if not np.isfinite(model.distance_map[start[0], start[1]]):
        return []

    if len(model.goal_idx) != 2:
        return []
    gx, gy = model.goal_idx
    actions_2d = model.actions

    path_idx = [start]
    visited = {start}
    cx, cy = start
    max_steps = max(300, min(2500, nx * nz * 2))

    for _ in range(max_steps):
        if (cx, cy) == (gx, gy):
            break

        a = int(model.policy[cx, cy])
        if a < 0 or a >= len(actions_2d):
            return []
        dx, dy = actions_2d[a]
        nx_i = cx + dx
        ny_i = cy + dy
        if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= nz:
            return []
        if not bool(model.free_mask[nx_i, ny_i]):
            return []

        cur_d = float(model.distance_map[cx, cy])
        next_d = float(model.distance_map[nx_i, ny_i])
        if not math.isfinite(next_d) or next_d >= cur_d:
            best = None
            best_d = float("inf")
            for bdx, bdy in actions_2d:
                bx = cx + bdx
                by = cy + bdy
                if bx < 0 or bx >= nx or by < 0 or by >= nz:
                    continue
                if not bool(model.free_mask[bx, by]):
                    continue
                bd = float(model.distance_map[bx, by])
                if not math.isfinite(bd):
                    continue
                if bd < best_d:
                    best_d = bd
                    best = (bx, by)
            if best is None or best_d >= cur_d:
                return []
            nx_i, ny_i = best

        node = (nx_i, ny_i)
        if node in visited:
            return []
        visited.add(node)
        path_idx.append(node)
        cx, cy = node

    if not path_idx or path_idx[-1] != (gx, gy):
        return []

    start_z = float(start_xyz[2])
    end_z = float(end_xyz[2])
    n = len(path_idx)

    planned: list[tuple[float, float, float]] = []
    for i, (ix, iy) in enumerate(path_idx):
        mx = float(ix) / float(nx - 1) * 100.0
        my = float(iy) / float(nz - 1) * 100.0
        t = 0.0 if n <= 1 else float(i) / float(n - 1)
        mz = (1.0 - t) * start_z + t * end_z
        planned.append((_clamp_0_100(mx), _clamp_0_100(my), _clamp_0_100(mz)))

    return planned
