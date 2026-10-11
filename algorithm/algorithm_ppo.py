"""
PPO training and path-building kernel implementation.

Creator: Orange JYJ
License: Apache-2.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from algorithm.algorithm_q_learning import train_q_learning_full_map
from algorithm.flight_constraints import FlightConstraints, build_flight_policy_path, copy_flight_context


@dataclass
class PPOModel:
    actor_w1: object
    actor_b1: object
    actor_w2: object
    actor_b2: object
    actor_w3: object
    actor_b3: object
    critic_w1: object
    critic_b1: object
    critic_w2: object
    critic_b2: object
    critic_w3: object
    critic_b3: object
    policy: object
    value: object
    free_mask: object
    goal_idx: tuple[int, ...]
    grid_shape: tuple[int, ...]
    is_3d: bool
    actions: tuple[tuple[int, ...], ...]
    distance_map: object
    flight_constraints: FlightConstraints | None = None
    flight_voxel_size: float = 1.0
    flight_valid_moves: object = None


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


def _actor_forward(model: PPOModel, x):
    h1 = _relu(x @ model.actor_w1 + model.actor_b1)
    h2 = _relu(h1 @ model.actor_w2 + model.actor_b2)
    logits = h2 @ model.actor_w3 + model.actor_b3
    return h1, h2, logits


def _critic_forward(model: PPOModel, x):
    h1 = _relu(x @ model.critic_w1 + model.critic_b1)
    h2 = _relu(h1 @ model.critic_w2 + model.critic_b2)
    v = h2 @ model.critic_w3 + model.critic_b3
    return h1, h2, v


def _nearest_free_from_model(model: PPOModel, start_idx):
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


def _best_descending_neighbor_3d(model: PPOModel, node, visited):
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


def _best_descending_neighbor_2d(model: PPOModel, node, visited):
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


def train_ppo_full_map(
    occupancy,
    heights,
    end_map_xyz: tuple[float, float, float] | None,
    map_to_index: Callable[[float, int, str], int],
    progress_callback: Callable[[int], None] | None = None,
    flight_constraints: FlightConstraints | None = None,
    voxel_size: float = 1.0,
) -> tuple[bool, str, PPOModel | None]:
    """Train PPO policy on full map.

    Uses a full-map teacher (global Q-learning solution), then performs
    PPO-style clipped policy optimization with a value head.
    """

    import numpy as np

    if end_map_xyz is None:
        return (False, "Set End point before PPO training", None)

    def teacher_progress(p: int) -> None:
        if progress_callback is not None:
            progress_callback(max(0, min(50, int(p * 0.5))))

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
        return (False, "PPO training failed: cannot build teacher targets", None)

    is_3d = bool(teacher_model.is_3d)
    actions = teacher_model.actions
    n_actions = len(actions)
    grid_shape = teacher_model.grid_shape
    free_mask = teacher_model.free_mask
    distance_map = teacher_model.distance_map

    free_indices = np.argwhere(free_mask)
    if free_indices.size == 0:
        return (False, "PPO training failed: no free states", None)

    inputs = _state_batch_from_indices(free_indices, grid_shape, is_3d)
    teacher_policy = teacher_model.policy[tuple(free_indices.T)].astype(np.int64)
    teacher_values = np.max(teacher_model.q_table[tuple(free_indices.T)], axis=1).astype(np.float32)

    # Advantage proxy from normalized teacher value.
    adv = teacher_values - float(np.mean(teacher_values))
    adv_std = float(np.std(adv)) + 1e-6
    adv = (adv / adv_std).astype(np.float32)

    state_dim = int(inputs.shape[1])
    hidden1 = 64
    hidden2 = 64
    rng = np.random.default_rng(11)

    actor_w1 = (rng.standard_normal((state_dim, hidden1)).astype(np.float32) * 0.08)
    actor_b1 = np.zeros((1, hidden1), dtype=np.float32)
    actor_w2 = (rng.standard_normal((hidden1, hidden2)).astype(np.float32) * 0.08)
    actor_b2 = np.zeros((1, hidden2), dtype=np.float32)
    actor_w3 = (rng.standard_normal((hidden2, n_actions)).astype(np.float32) * 0.08)
    actor_b3 = np.zeros((1, n_actions), dtype=np.float32)

    critic_w1 = (rng.standard_normal((state_dim, hidden1)).astype(np.float32) * 0.08)
    critic_b1 = np.zeros((1, hidden1), dtype=np.float32)
    critic_w2 = (rng.standard_normal((hidden1, hidden2)).astype(np.float32) * 0.08)
    critic_b2 = np.zeros((1, hidden2), dtype=np.float32)
    critic_w3 = (rng.standard_normal((hidden2, 1)).astype(np.float32) * 0.08)
    critic_b3 = np.zeros((1, 1), dtype=np.float32)

    model = PPOModel(
        actor_w1=actor_w1,
        actor_b1=actor_b1,
        actor_w2=actor_w2,
        actor_b2=actor_b2,
        actor_w3=actor_w3,
        actor_b3=actor_b3,
        critic_w1=critic_w1,
        critic_b1=critic_b1,
        critic_w2=critic_w2,
        critic_b2=critic_b2,
        critic_w3=critic_w3,
        critic_b3=critic_b3,
        policy=None,
        value=None,
        free_mask=free_mask,
        goal_idx=teacher_model.goal_idx,
        grid_shape=grid_shape,
        is_3d=is_3d,
        actions=actions,
        distance_map=distance_map,
    )

    # Snapshot old policy for PPO ratio.
    _, _, old_logits = _actor_forward(model, inputs)
    old_probs = _softmax(old_logits)
    old_pi_a = old_probs[np.arange(len(inputs)), teacher_policy]

    epochs = 120
    batch = min(2048, int(inputs.shape[0]))
    lr_actor = 0.01
    lr_critic = 0.01
    clip_eps = 0.2
    ent_coef = 0.01

    idx_all = np.arange(inputs.shape[0])

    for ep in range(epochs):
        sel = rng.choice(idx_all, size=batch, replace=batch > len(idx_all))
        x = inputs[sel]
        a_idx = teacher_policy[sel]
        adv_b = adv[sel]
        old_pi = np.maximum(1e-8, old_pi_a[sel])
        v_target = teacher_values[sel].reshape(-1, 1)

        # Actor forward.
        h1, h2, logits = _actor_forward(model, x)
        probs = _softmax(logits)
        pi_a = np.maximum(1e-8, probs[np.arange(len(x)), a_idx])
        ratio = pi_a / old_pi
        ratio_clip = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

        # Chosen surrogate branch per sample.
        s1 = ratio * adv_b
        s2 = ratio_clip * adv_b
        use_clip = np.where(adv_b >= 0.0, ratio > ratio_clip, ratio < ratio_clip)
        w = np.where(use_clip, ratio_clip, ratio) * adv_b

        # Gradient for actor from weighted log-prob and entropy bonus.
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(x)), a_idx] = 1.0
        dlogits = -(w[:, None] / max(1, len(x))) * (onehot - probs)
        dlogits += ent_coef * (probs * (np.log(np.maximum(1e-8, probs)) + 1.0)) / max(1, len(x))

        gw3 = h2.T @ dlogits
        gb3 = np.sum(dlogits, axis=0, keepdims=True)
        dh2 = dlogits @ model.actor_w3.T
        dh2[h2 <= 0.0] = 0.0
        gw2 = h1.T @ dh2
        gb2 = np.sum(dh2, axis=0, keepdims=True)
        dh1 = dh2 @ model.actor_w2.T
        dh1[h1 <= 0.0] = 0.0
        gw1 = x.T @ dh1
        gb1 = np.sum(dh1, axis=0, keepdims=True)

        model.actor_w3 = (model.actor_w3 - lr_actor * gw3).astype(np.float32)
        model.actor_b3 = (model.actor_b3 - lr_actor * gb3).astype(np.float32)
        model.actor_w2 = (model.actor_w2 - lr_actor * gw2).astype(np.float32)
        model.actor_b2 = (model.actor_b2 - lr_actor * gb2).astype(np.float32)
        model.actor_w1 = (model.actor_w1 - lr_actor * gw1).astype(np.float32)
        model.actor_b1 = (model.actor_b1 - lr_actor * gb1).astype(np.float32)

        # Critic update (MSE to teacher value).
        ch1, ch2, v_pred = _critic_forward(model, x)
        dv = 2.0 * (v_pred - v_target) / max(1, len(x))

        cgw3 = ch2.T @ dv
        cgb3 = np.sum(dv, axis=0, keepdims=True)
        cdh2 = dv @ model.critic_w3.T
        cdh2[ch2 <= 0.0] = 0.0
        cgw2 = ch1.T @ cdh2
        cgb2 = np.sum(cdh2, axis=0, keepdims=True)
        cdh1 = cdh2 @ model.critic_w2.T
        cdh1[ch1 <= 0.0] = 0.0
        cgw1 = x.T @ cdh1
        cgb1 = np.sum(cdh1, axis=0, keepdims=True)

        model.critic_w3 = (model.critic_w3 - lr_critic * cgw3).astype(np.float32)
        model.critic_b3 = (model.critic_b3 - lr_critic * cgb3).astype(np.float32)
        model.critic_w2 = (model.critic_w2 - lr_critic * cgw2).astype(np.float32)
        model.critic_b2 = (model.critic_b2 - lr_critic * cgb2).astype(np.float32)
        model.critic_w1 = (model.critic_w1 - lr_critic * cgw1).astype(np.float32)
        model.critic_b1 = (model.critic_b1 - lr_critic * cgb1).astype(np.float32)

        if progress_callback is not None and ((ep + 1) % max(1, epochs // 40) == 0 or ep + 1 == epochs):
            p = 50 + int((ep + 1) * 50 / epochs)
            progress_callback(max(0, min(100, p)))

    # Materialize dense policy/value for fast navigation.
    _, _, logits_all = _actor_forward(model, inputs)
    probs_all = _softmax(logits_all)
    pred_policy = np.argmax(probs_all, axis=1).astype(np.int8)

    _, _, v_all = _critic_forward(model, inputs)
    v_all = v_all.reshape(-1).astype(np.float32)

    if is_3d:
        nx, ny_h, nz = grid_shape
        dense_policy = np.zeros((nx, ny_h, nz), dtype=np.int8)
        dense_value = np.full((nx, ny_h, nz), -1e9, dtype=np.float32)
        for i, (ix, iy_h, iz) in enumerate(free_indices):
            dense_policy[int(ix), int(iy_h), int(iz)] = pred_policy[i]
            dense_value[int(ix), int(iy_h), int(iz)] = v_all[i]
    else:
        nx, ny = grid_shape
        dense_policy = np.zeros((nx, ny), dtype=np.int8)
        dense_value = np.full((nx, ny), -1e9, dtype=np.float32)
        for i, (ix, iy) in enumerate(free_indices):
            dense_policy[int(ix), int(iy)] = pred_policy[i]
            dense_value[int(ix), int(iy)] = v_all[i]

    model.policy = dense_policy
    model.value = dense_value

    copy_flight_context(model, teacher_model)
    if progress_callback is not None:
        progress_callback(100)

    return (
        True,
        f"PPO trained: teacher_steps={episodes}, free={free_cells}, params=({state_dim},{hidden1},{hidden2},{n_actions})",
        model,
    )


def build_ppo_path(
    model: PPOModel,
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    map_to_index: Callable[[float, int, str], int],
) -> list[tuple[float, float, float]]:
    if model.flight_constraints is not None:
        import numpy as np
        def scores(node):
            inputs = _state_batch_from_indices(np.asarray([node]), model.grid_shape, True)
            return _actor_forward(model, inputs)[2][0]
        return build_flight_policy_path(model, start_xyz, end_xyz, scores)
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
