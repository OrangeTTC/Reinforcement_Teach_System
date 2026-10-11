"""
HER training and path-building kernel implementation.

Creator: Orange JYJ
License: Apache-2.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from algorithm.algorithm_q_learning import train_q_learning_full_map
from algorithm.flight_constraints import FlightConstraints, build_flight_policy_path, copy_flight_context
from algorithm.flight_speed_control import train_speed_controller


@dataclass
class HERModel:
    w1: object
    b1: object
    w2: object
    b2: object
    w3: object
    b3: object
    free_mask: object
    goal_idx: tuple[int, ...]
    grid_shape: tuple[int, ...]
    is_3d: bool
    actions: tuple[tuple[int, ...], ...]
    distance_map: object
    flight_constraints: FlightConstraints | None = None
    flight_voxel_size: float = 1.0
    flight_valid_moves: object = None
    speed_controller: object = None


def _clamp_0_100(v: float) -> float:
    return max(0.0, min(100.0, float(v)))


def _relu(x):
    import numpy as np

    return np.maximum(x, 0.0)


def _softmax(logits):
    import numpy as np

    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(1e-9, e.sum(axis=1, keepdims=True))


def _forward(model: HERModel, x):
    h1 = _relu(x @ model.w1 + model.b1)
    h2 = _relu(h1 @ model.w2 + model.b2)
    logits = h2 @ model.w3 + model.b3
    return h1, h2, logits


def _state_goal_features(states, goals, grid_shape, is_3d: bool):
    import numpy as np

    s = np.asarray(states, dtype=np.float32)
    g = np.asarray(goals, dtype=np.float32)

    if is_3d:
        nx, ny_h, nz = grid_shape
        sx = s[:, 0] / max(1.0, float(nx - 1))
        sy_h = s[:, 1] / max(1.0, float(ny_h - 1))
        sz = s[:, 2] / max(1.0, float(nz - 1))
        gx = g[:, 0] / max(1.0, float(nx - 1))
        gy_h = g[:, 1] / max(1.0, float(ny_h - 1))
        gz = g[:, 2] / max(1.0, float(nz - 1))
        return np.stack([sx, sy_h, sz, gx, gy_h, gz], axis=1)

    nx, ny = grid_shape
    sx = s[:, 0] / max(1.0, float(nx - 1))
    sy = s[:, 1] / max(1.0, float(ny - 1))
    gx = g[:, 0] / max(1.0, float(nx - 1))
    gy = g[:, 1] / max(1.0, float(ny - 1))
    return np.stack([sx, sy, gx, gy], axis=1)


def _best_descending_neighbor_3d(model: HERModel, node, visited):
    cx, cy_h, cz = node
    nx, ny_h, nz = model.grid_shape
    cur_d = float(model.distance_map[cx, cy_h, cz])
    best = None
    best_d = float("inf")
    for bdx, bdy_h, bdz in model.actions:
        bx = cx + bdx
        by = cy_h + bdy_h
        bz = cz + bdz
        if bx < 0 or bx >= nx or by < 0 or by >= ny_h or bz < 0 or bz >= nz:
            continue
        if not bool(model.free_mask[bx, by, bz]) or (bx, by, bz) in visited:
            continue
        bd = float(model.distance_map[bx, by, bz])
        if not math.isfinite(bd):
            continue
        if bd < best_d:
            best_d = bd
            best = (bx, by, bz)
    if best is None or best_d >= cur_d:
        return None
    return best


def _best_descending_neighbor_2d(model: HERModel, node, visited):
    cx, cy = node
    nx, ny = model.grid_shape
    cur_d = float(model.distance_map[cx, cy])
    best = None
    best_d = float("inf")
    for bdx, bdy in model.actions:
        bx = cx + bdx
        by = cy + bdy
        if bx < 0 or bx >= nx or by < 0 or by >= ny:
            continue
        if not bool(model.free_mask[bx, by]) or (bx, by) in visited:
            continue
        bd = float(model.distance_map[bx, by])
        if not math.isfinite(bd):
            continue
        if bd < best_d:
            best_d = bd
            best = (bx, by)
    if best is None or best_d >= cur_d:
        return None
    return best


def _nearest_free_from_model(model: HERModel, start_idx):
    if model.is_3d:
        nx, ny_h, nz = model.grid_shape
        sx, sy_h, sz = start_idx
        sx = max(0, min(nx - 1, int(sx)))
        sy_h = max(0, min(ny_h - 1, int(sy_h)))
        sz = max(0, min(nz - 1, int(sz)))
        if bool(model.free_mask[sx, sy_h, sz]):
            return (sx, sy_h, sz)

        queue = [(sx, sy_h, sz)]
        visited = {(sx, sy_h, sz)}
        head = 0
        neigh = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        while head < len(queue):
            cx, cy_h, cz = queue[head]
            head += 1
            for dx, dy_h, dz in neigh:
                nx_i = cx + dx
                ny_i = cy_h + dy_h
                nz_i = cz + dz
                if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny_h or nz_i < 0 or nz_i >= nz:
                    continue
                node = (nx_i, ny_i, nz_i)
                if node in visited:
                    continue
                if bool(model.free_mask[nx_i, ny_i, nz_i]):
                    return node
                visited.add(node)
                queue.append(node)
        return None

    nx, ny = model.grid_shape
    sx, sy = start_idx
    sx = max(0, min(nx - 1, int(sx)))
    sy = max(0, min(ny - 1, int(sy)))
    if bool(model.free_mask[sx, sy]):
        return (sx, sy)

    queue = [(sx, sy)]
    visited = {(sx, sy)}
    head = 0
    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while head < len(queue):
        cx, cy = queue[head]
        head += 1
        for dx, dy in neigh:
            nx_i = cx + dx
            ny_i = cy + dy
            if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny:
                continue
            node = (nx_i, ny_i)
            if node in visited:
                continue
            if bool(model.free_mask[nx_i, ny_i]):
                return node
            visited.add(node)
            queue.append(node)
    return None


def _teacher_step(model, node, visited):
    if getattr(model, 'flight_constraints', None) is not None:
        import numpy as np
        best, best_score = None, -math.inf
        for a in np.flatnonzero(model.flight_valid_moves[(slice(None),) + node]):
            nxt = tuple(c + d for c, d in zip(node, model.actions[a]))
            if nxt in visited or model.distance_map[nxt] >= model.distance_map[node] - 1e-9:
                continue
            score = float(model.q_table[node + (a,)])
            if score > best_score:
                best, best_score = (nxt, int(a)), score
        return best
    if model.is_3d:
        cx, cy_h, cz = node
        nx, ny_h, nz = model.grid_shape
        a = int(model.policy[cx, cy_h, cz])
        if 0 <= a < len(model.actions):
            dx, dy_h, dz = model.actions[a]
            nx_i = cx + dx
            ny_i = cy_h + dy_h
            nz_i = cz + dz
            if (
                0 <= nx_i < nx
                and 0 <= ny_i < ny_h
                and 0 <= nz_i < nz
                and bool(model.free_mask[nx_i, ny_i, nz_i])
                and (nx_i, ny_i, nz_i) not in visited
            ):
                cur_d = float(model.distance_map[cx, cy_h, cz])
                nd = float(model.distance_map[nx_i, ny_i, nz_i])
                if math.isfinite(nd) and nd < cur_d:
                    return (nx_i, ny_i, nz_i), a

        best = _best_descending_neighbor_3d(model, node, visited)
        if best is None:
            return None
        bx, by, bz = best
        for idx, (dx, dy_h, dz) in enumerate(model.actions):
            if (cx + dx, cy_h + dy_h, cz + dz) == (bx, by, bz):
                return best, idx
        return None

    cx, cy = node
    nx, ny = model.grid_shape
    a = int(model.policy[cx, cy])
    if 0 <= a < len(model.actions):
        dx, dy = model.actions[a]
        nx_i = cx + dx
        ny_i = cy + dy
        if (
            0 <= nx_i < nx
            and 0 <= ny_i < ny
            and bool(model.free_mask[nx_i, ny_i])
            and (nx_i, ny_i) not in visited
        ):
            cur_d = float(model.distance_map[cx, cy])
            nd = float(model.distance_map[nx_i, ny_i])
            if math.isfinite(nd) and nd < cur_d:
                return (nx_i, ny_i), a

    best = _best_descending_neighbor_2d(model, node, visited)
    if best is None:
        return None
    bx, by = best
    for idx, (dx, dy) in enumerate(model.actions):
        if (cx + dx, cy + dy) == (bx, by):
            return best, idx
    return None


def train_her_full_map(
    occupancy,
    heights,
    end_map_xyz: tuple[float, float, float] | None,
    map_to_index: Callable[[float, int, str], int],
    progress_callback: Callable[[int], None] | None = None,
    flight_constraints: FlightConstraints | None = None,
    voxel_size: float = 1.0,
) -> tuple[bool, str, HERModel | None]:
    """Train HER policy on full map with hindsight relabeling."""

    import numpy as np

    if end_map_xyz is None:
        return (False, "Set End point before HER training", None)

    def teacher_progress(p: int) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(45, int(p * 0.45))))

    ok, teacher_model, episodes, free_cells, _ = train_q_learning_full_map(
        occupancy=occupancy,
        heights=heights,
        end_map_xyz=end_map_xyz,
        map_to_index=map_to_index,
        progress_callback=teacher_progress,
        flight_constraints=flight_constraints,
        voxel_size=voxel_size,
    )
    if not ok or teacher_model is None:
        return (False, "HER training failed: cannot build teacher targets", None)

    is_3d = bool(teacher_model.is_3d)
    actions = teacher_model.actions
    n_actions = len(actions)
    grid_shape = teacher_model.grid_shape
    free_mask = teacher_model.free_mask
    goal_idx = teacher_model.goal_idx
    distance_map = teacher_model.distance_map

    free_states = np.argwhere(free_mask)
    if free_states.size == 0:
        return (False, "HER training failed: no free states", None)

    # Build a lightweight teacher wrapper for step generation.
    teacher_wrap = type("TeacherWrap", (), {})()
    teacher_wrap.is_3d = is_3d
    teacher_wrap.grid_shape = grid_shape
    teacher_wrap.free_mask = free_mask
    teacher_wrap.goal_idx = goal_idx
    teacher_wrap.distance_map = distance_map
    teacher_wrap.actions = actions
    teacher_wrap.policy = teacher_model.policy
    teacher_wrap.q_table = teacher_model.q_table
    copy_flight_context(teacher_wrap, teacher_model)

    # Base dataset: all states with true goal and teacher action.
    states = []
    goals = []
    labels = []
    for s in free_states:
        s_t = tuple(int(v) for v in s)
        visited = {s_t}
        step = _teacher_step(teacher_wrap, s_t, visited)
        if step is None:
            continue
        nxt, a = step
        _ = nxt
        states.append(s_t)
        goals.append(goal_idx)
        labels.append(int(a))

    # HER augmentation from teacher trajectories.
    rng = np.random.default_rng(23)
    traj_count = min(800, max(150, len(states) // 8))
    max_steps = 80 if is_3d else 50

    for i in range(traj_count):
        start = tuple(int(v) for v in free_states[rng.integers(0, len(free_states))])
        if tuple(start) == tuple(goal_idx):
            continue
        path = [start]
        acts = []
        visited = {start}
        cur = start
        for _ in range(max_steps):
            if cur == tuple(goal_idx):
                break
            step = _teacher_step(teacher_wrap, cur, visited)
            if step is None:
                break
            nxt, a = step
            acts.append(int(a))
            path.append(nxt)
            visited.add(nxt)
            cur = nxt

        if len(path) < 3:
            continue

        for t in range(len(path) - 2):
            s = path[t]
            a = acts[t]

            # Original goal sample.
            states.append(s)
            goals.append(goal_idx)
            labels.append(a)

            # Hindsight relabel: choose a future achieved state as goal.
            hi = min(len(path) - 1, t + 8)
            k = int(rng.integers(t + 1, hi + 1))
            hg = path[k]
            states.append(s)
            goals.append(hg)
            labels.append(a)

        if progress_callback is not None and (i + 1) % max(1, traj_count // 20) == 0:
            progress_callback(45 + int((i + 1) * 20 / traj_count))

    if not states:
        return (False, "HER training failed: empty dataset", None)

    x = _state_goal_features(states, goals, grid_shape, is_3d)
    y = np.asarray(labels, dtype=np.int64)

    in_dim = int(x.shape[1])
    h1_dim = 64
    h2_dim = 64
    w1 = (rng.standard_normal((in_dim, h1_dim)).astype(np.float32) * 0.08)
    b1 = np.zeros((1, h1_dim), dtype=np.float32)
    w2 = (rng.standard_normal((h1_dim, h2_dim)).astype(np.float32) * 0.08)
    b2 = np.zeros((1, h2_dim), dtype=np.float32)
    w3 = (rng.standard_normal((h2_dim, n_actions)).astype(np.float32) * 0.08)
    b3 = np.zeros((1, n_actions), dtype=np.float32)

    model = HERModel(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        w3=w3,
        b3=b3,
        free_mask=free_mask,
        goal_idx=goal_idx,
        grid_shape=grid_shape,
        is_3d=is_3d,
        actions=actions,
        distance_map=distance_map,
    )

    epochs = 120
    batch = min(2048, int(x.shape[0]))
    lr = 0.015
    idx_all = np.arange(x.shape[0])

    for ep in range(epochs):
        sel = rng.choice(idx_all, size=batch, replace=batch > len(idx_all))
        xb = x[sel]
        yb = y[sel]

        h1, h2, logits = _forward(model, xb)
        probs = _softmax(logits)
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(xb)), yb] = 1.0
        dlogits = (probs - onehot) / max(1, len(xb))

        gw3 = h2.T @ dlogits
        gb3 = np.sum(dlogits, axis=0, keepdims=True)
        dh2 = dlogits @ model.w3.T
        dh2[h2 <= 0.0] = 0.0
        gw2 = h1.T @ dh2
        gb2 = np.sum(dh2, axis=0, keepdims=True)
        dh1 = dh2 @ model.w2.T
        dh1[h1 <= 0.0] = 0.0
        gw1 = xb.T @ dh1
        gb1 = np.sum(dh1, axis=0, keepdims=True)

        model.w3 = (model.w3 - lr * gw3).astype(np.float32)
        model.b3 = (model.b3 - lr * gb3).astype(np.float32)
        model.w2 = (model.w2 - lr * gw2).astype(np.float32)
        model.b2 = (model.b2 - lr * gb2).astype(np.float32)
        model.w1 = (model.w1 - lr * gw1).astype(np.float32)
        model.b1 = (model.b1 - lr * gb1).astype(np.float32)

        if progress_callback is not None and ((ep + 1) % max(1, epochs // 35) == 0 or ep + 1 == epochs):
            progress_callback(65 + int((ep + 1) * 35 / epochs))

    copy_flight_context(model, teacher_model)
    if flight_constraints is not None:
        model.speed_controller = train_speed_controller('HER', flight_constraints)
    if progress_callback is not None:
        progress_callback(100)

    return (
        True,
        f"HER trained: teacher_steps={episodes}, free={free_cells}, dataset={len(states)}",
        model,
    )


def build_her_path(
    model: HERModel,
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    map_to_index: Callable[[float, int, str], int],
) -> list[tuple[float, float, float]]:
    if model.flight_constraints is not None:
        def scores(node):
            inputs = _state_goal_features([node], [model.goal_idx], model.grid_shape, True)
            return _forward(model, inputs)[2][0]
        return build_flight_policy_path(model, start_xyz, end_xyz, scores)
    import numpy as np

    _ = end_xyz

    goal_idx = model.goal_idx

    if model.is_3d:
        nx, ny_h, nz = model.grid_shape
        sx = map_to_index(start_xyz[0], nx, "round")
        sy_plane = map_to_index(start_xyz[1], nz, "round")
        sz_h = map_to_index(start_xyz[2], ny_h, "round")
        start = _nearest_free_from_model(model, (sx, sz_h, sy_plane))
        if start is None:
            return []
        if not np.isfinite(model.distance_map[start[0], start[1], start[2]]):
            return []

        gx, gy_h, gz = goal_idx
        path_idx = [start]
        visited = {start}
        cx, cy_h, cz = start
        max_steps = max(1000, min(12000, nx * ny_h * nz // 2))

        for _ in range(max_steps):
            if (cx, cy_h, cz) == (gx, gy_h, gz):
                break

            feat = _state_goal_features([[cx, cy_h, cz]], [[gx, gy_h, gz]], model.grid_shape, True)
            _, _, logits = _forward(model, feat)
            a = int(np.argmax(logits[0]))

            proposed = None
            if 0 <= a < len(model.actions):
                dx, dy_h, dz = model.actions[a]
                nx_i = cx + dx
                ny_i = cy_h + dy_h
                nz_i = cz + dz
                if (
                    0 <= nx_i < nx
                    and 0 <= ny_i < ny_h
                    and 0 <= nz_i < nz
                    and bool(model.free_mask[nx_i, ny_i, nz_i])
                    and (nx_i, ny_i, nz_i) not in visited
                ):
                    cur_d = float(model.distance_map[cx, cy_h, cz])
                    next_d = float(model.distance_map[nx_i, ny_i, nz_i])
                    if math.isfinite(next_d) and next_d < cur_d:
                        proposed = (nx_i, ny_i, nz_i)

            if proposed is None:
                proposed = _best_descending_neighbor_3d(model, (cx, cy_h, cz), visited)
            if proposed is None:
                return []

            nx_i, ny_i, nz_i = proposed
            node = (nx_i, ny_i, nz_i)
            if node in visited:
                return []
            visited.add(node)
            path_idx.append(node)
            cx, cy_h, cz = node

        if not path_idx or path_idx[-1] != (gx, gy_h, gz):
            return []

        planned = []
        for ix, iy_h, iz in path_idx:
            mx = float(ix) / float(nx - 1) * 100.0
            my = float(iz) / float(nz - 1) * 100.0
            mz = float(iy_h) / float(ny_h - 1) * 100.0
            planned.append((_clamp_0_100(mx), _clamp_0_100(my), _clamp_0_100(mz)))
        return planned

    nx, ny = model.grid_shape
    sx = map_to_index(start_xyz[0], nx, "round")
    sy = map_to_index(start_xyz[1], ny, "round")
    start = _nearest_free_from_model(model, (sx, sy))
    if start is None:
        return []
    if not np.isfinite(model.distance_map[start[0], start[1]]):
        return []

    gx, gy = goal_idx
    path_idx = [start]
    visited = {start}
    cx, cy = start
    max_steps = max(300, min(2500, nx * ny * 2))

    for _ in range(max_steps):
        if (cx, cy) == (gx, gy):
            break

        feat = _state_goal_features([[cx, cy]], [[gx, gy]], model.grid_shape, False)
        _, _, logits = _forward(model, feat)
        a = int(np.argmax(logits[0]))

        proposed = None
        if 0 <= a < len(model.actions):
            dx, dy = model.actions[a]
            nx_i = cx + dx
            ny_i = cy + dy
            if (
                0 <= nx_i < nx
                and 0 <= ny_i < ny
                and bool(model.free_mask[nx_i, ny_i])
                and (nx_i, ny_i) not in visited
            ):
                cur_d = float(model.distance_map[cx, cy])
                next_d = float(model.distance_map[nx_i, ny_i])
                if math.isfinite(next_d) and next_d < cur_d:
                    proposed = (nx_i, ny_i)

        if proposed is None:
            proposed = _best_descending_neighbor_2d(model, (cx, cy), visited)
        if proposed is None:
            return []

        nx_i, ny_i = proposed
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
    planned = []
    for i, (ix, iy) in enumerate(path_idx):
        mx = float(ix) / float(nx - 1) * 100.0
        my = float(iy) / float(ny - 1) * 100.0
        t = 0.0 if n <= 1 else float(i) / float(n - 1)
        mz = (1.0 - t) * start_z + t * end_z
        planned.append((_clamp_0_100(mx), _clamp_0_100(my), _clamp_0_100(mz)))
    return planned
