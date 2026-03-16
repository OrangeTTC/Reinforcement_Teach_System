from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from q_learning import QLearningModel, train_q_learning_full_map


@dataclass
class DQNModel:
    w1: object
    b1: object
    w2: object
    b2: object
    w3: object
    b3: object
    policy: object
    q_table: object
    free_mask: object
    goal_idx: tuple[int, ...]
    grid_shape: tuple[int, ...]
    is_3d: bool
    actions: tuple[tuple[int, ...], ...]
    distance_map: object


def _relu(x):
    import numpy as np

    return np.maximum(x, 0.0)


def _forward(model: DQNModel, x):
    h1 = _relu(x @ model.w1 + model.b1)
    h2 = _relu(h1 @ model.w2 + model.b2)
    return h1, h2, h2 @ model.w3 + model.b3


def _predict_q(model: DQNModel, x):
    _, _, y = _forward(model, x)
    return y


def _best_descending_neighbor_3d(model: DQNModel, node, visited):
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
        if not bool(model.free_mask[bx, by, bz]):
            continue
        if (bx, by, bz) in visited:
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


def _best_descending_neighbor_2d(model: DQNModel, node, visited):
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
        if not bool(model.free_mask[bx, by]):
            continue
        if (bx, by) in visited:
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


def _nearest_free_from_model(model: DQNModel, start_idx):
    import numpy as np

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


def _state_batch_from_indices(indices, grid_shape, is_3d: bool):
    import numpy as np

    arr = np.asarray(indices, dtype=np.float32)
    if is_3d:
        nx, ny_h, nz = grid_shape
        sx = arr[:, 0] / max(1.0, float(nx - 1))
        sy_h = arr[:, 1] / max(1.0, float(ny_h - 1))
        sz = arr[:, 2] / max(1.0, float(nz - 1))
        return np.stack([sx, sy_h, sz], axis=1)

    nx, ny = grid_shape
    sx = arr[:, 0] / max(1.0, float(nx - 1))
    sy = arr[:, 1] / max(1.0, float(ny - 1))
    return np.stack([sx, sy], axis=1)


def train_dqn_full_map(
    occupancy,
    heights,
    end_map_xyz: tuple[float, float, float] | None,
    map_to_index: Callable[[float, int, str], int],
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[bool, str, DQNModel | None]:
    """Train DQN policy on full map.

    Implementation uses full-map Q targets (teacher from global Q-learning) and
    distills them into a compact neural approximator for DQN-like inference.
    """

    import numpy as np

    if end_map_xyz is None:
        return (False, "Set End point before DQN training", None)

    def teacher_progress(p: int) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(60, int(p * 0.6))))

    ok, teacher_model, episodes, free_cells, _ = train_q_learning_full_map(
        occupancy=occupancy,
        heights=heights,
        end_map_xyz=end_map_xyz,
        map_to_index=map_to_index,
        progress_callback=teacher_progress,
    )
    if not ok or teacher_model is None:
        return (False, "DQN training failed: cannot build teacher targets", None)

    is_3d = bool(teacher_model.is_3d)
    actions = teacher_model.actions
    n_actions = len(actions)
    grid_shape = teacher_model.grid_shape
    free_mask = teacher_model.free_mask
    distance_map = teacher_model.distance_map

    free_indices = np.argwhere(free_mask)
    if free_indices.size == 0:
        return (False, "DQN training failed: no free states", None)

    targets = teacher_model.q_table[tuple(free_indices.T)]
    targets = np.clip(targets, -300.0, 300.0)
    inputs = _state_batch_from_indices(free_indices, grid_shape, is_3d)

    state_dim = int(inputs.shape[1])
    hidden1 = 64
    hidden2 = 64
    rng = np.random.default_rng(7)

    w1 = (rng.standard_normal((state_dim, hidden1)).astype(np.float32) * 0.08)
    b1 = np.zeros((1, hidden1), dtype=np.float32)
    w2 = (rng.standard_normal((hidden1, hidden2)).astype(np.float32) * 0.08)
    b2 = np.zeros((1, hidden2), dtype=np.float32)
    w3 = (rng.standard_normal((hidden2, n_actions)).astype(np.float32) * 0.08)
    b3 = np.zeros((1, n_actions), dtype=np.float32)

    temp_model = DQNModel(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        w3=w3,
        b3=b3,
        policy=None,
        q_table=None,
        free_mask=free_mask,
        goal_idx=teacher_model.goal_idx,
        grid_shape=grid_shape,
        is_3d=is_3d,
        actions=actions,
        distance_map=distance_map,
    )

    epochs = 140
    batch = min(2048, int(inputs.shape[0]))
    lr = 0.02

    idx_all = np.arange(inputs.shape[0])

    for ep in range(epochs):
        sel = rng.choice(idx_all, size=batch, replace=batch > len(idx_all))
        x = inputs[sel]
        y_true = targets[sel]

        h1, h2, y_pred = _forward(temp_model, x)
        err = (y_pred - y_true).astype(np.float32)
        scale = float(1.0 / max(1, batch))

        # MSE gradients.
        dy = 2.0 * err * scale
        gw3 = h2.T @ dy
        gb3 = np.sum(dy, axis=0, keepdims=True)

        dh2 = dy @ temp_model.w3.T
        dh2[h2 <= 0.0] = 0.0
        gw2 = h1.T @ dh2
        gb2 = np.sum(dh2, axis=0, keepdims=True)

        dh1 = dh2 @ temp_model.w2.T
        dh1[h1 <= 0.0] = 0.0
        gw1 = x.T @ dh1
        gb1 = np.sum(dh1, axis=0, keepdims=True)

        temp_model.w3 = (temp_model.w3 - lr * gw3).astype(np.float32)
        temp_model.b3 = (temp_model.b3 - lr * gb3).astype(np.float32)
        temp_model.w2 = (temp_model.w2 - lr * gw2).astype(np.float32)
        temp_model.b2 = (temp_model.b2 - lr * gb2).astype(np.float32)
        temp_model.w1 = (temp_model.w1 - lr * gw1).astype(np.float32)
        temp_model.b1 = (temp_model.b1 - lr * gb1).astype(np.float32)

        if progress_callback is not None and ((ep + 1) % max(1, epochs // 40) == 0 or ep + 1 == epochs):
            p = 60 + int((ep + 1) * 40 / epochs)
            progress_callback(max(0, min(100, p)))

    pred_q = _predict_q(temp_model, inputs)
    pred_policy = np.argmax(pred_q, axis=1).astype(np.int8)

    # Build dense policy and q-table for fast runtime lookup.
    if is_3d:
        nx, ny_h, nz = grid_shape
        dense_policy = np.zeros((nx, ny_h, nz), dtype=np.int8)
        dense_q = np.full((nx, ny_h, nz, n_actions), -1e9, dtype=np.float32)
        for i, (ix, iy_h, iz) in enumerate(free_indices):
            dense_policy[int(ix), int(iy_h), int(iz)] = pred_policy[i]
            dense_q[int(ix), int(iy_h), int(iz), :] = pred_q[i]
    else:
        nx, ny = grid_shape
        dense_policy = np.zeros((nx, ny), dtype=np.int8)
        dense_q = np.full((nx, ny, n_actions), -1e9, dtype=np.float32)
        for i, (ix, iy) in enumerate(free_indices):
            dense_policy[int(ix), int(iy)] = pred_policy[i]
            dense_q[int(ix), int(iy), :] = pred_q[i]

    model = DQNModel(
        w1=temp_model.w1,
        b1=temp_model.b1,
        w2=temp_model.w2,
        b2=temp_model.b2,
        w3=temp_model.w3,
        b3=temp_model.b3,
        policy=dense_policy,
        q_table=dense_q,
        free_mask=free_mask,
        goal_idx=teacher_model.goal_idx,
        grid_shape=grid_shape,
        is_3d=is_3d,
        actions=actions,
        distance_map=distance_map,
    )

    if progress_callback is not None:
        progress_callback(100)

    return (
        True,
        f"DQN trained: teacher_steps={episodes}, free={free_cells}, params=({state_dim},{hidden1},{hidden2},{n_actions})",
        model,
    )


def build_dqn_path(
    model: DQNModel,
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    map_to_index: Callable[[float, int, str], int],
) -> list[tuple[float, float, float]]:
    import numpy as np

    _ = end_xyz

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

        gx, gy_h, gz = model.goal_idx
        path_idx = [start]
        visited = {start}
        cx, cy_h, cz = start
        max_steps = max(1000, min(12000, nx * ny_h * nz // 2))

        for _ in range(max_steps):
            if (cx, cy_h, cz) == (gx, gy_h, gz):
                break

            a = int(model.policy[cx, cy_h, cz])
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

    gx, gy = model.goal_idx
    path_idx = [start]
    visited = {start}
    cx, cy = start
    max_steps = max(300, min(2500, nx * ny * 2))

    for _ in range(max_steps):
        if (cx, cy) == (gx, gy):
            break

        a = int(model.policy[cx, cy])
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
    planned: list[tuple[float, float, float]] = []
    for i, (ix, iy) in enumerate(path_idx):
        mx = float(ix) / float(nx - 1) * 100.0
        my = float(iy) / float(ny - 1) * 100.0
        t = 0.0 if n <= 1 else float(i) / float(n - 1)
        mz = (1.0 - t) * start_z + t * end_z
        planned.append((_clamp_0_100(mx), _clamp_0_100(my), _clamp_0_100(mz)))
    return planned


def _clamp_0_100(v: float) -> float:
    return max(0.0, min(100.0, float(v)))
