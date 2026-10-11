"""Optional flight performance, planning geometry and analytic trajectories.

Creator: Orange JYJ
License: Apache-2.0
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np

MapPoint = tuple[float, float, float]
GridPoint = tuple[int, int, int]


@dataclass(frozen=True)
class FlightConstraints:
    """SI units; voxel_size is interpreted as metres per voxel.

    Cruise speed is maintained after the initial longitudinal acceleration.
    Climb_rate limits the magnitude of both ascent and descent velocity.
    Penalties are deliberately small, soft preferences during policy extraction.
    """

    max_speed: float = 12.0
    cruise_speed: float = 5.0
    longitudinal_accel: float = 2.0
    lateral_accel: float = 8.0
    climb_rate: float = 2.0
    turn_penalty: float = 0.15
    close_turn_penalty: float = 0.30

    def validate(self) -> None:
        values = (self.max_speed, self.cruise_speed, self.longitudinal_accel,
                  self.lateral_accel, self.climb_rate)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("速度、加速度和爬升速率必须为有限正数")
        if self.cruise_speed > self.max_speed:
            raise ValueError("巡航速度不能超过飞机最大速度")
        if not all(math.isfinite(v) and v >= 0 for v in (self.turn_penalty, self.close_turn_penalty)):
            raise ValueError("转弯惩罚必须为有限非负数")

    @property
    def turn_radius(self) -> float:
        return self.cruise_speed ** 2 / self.lateral_accel

    @property
    def vertical_fraction(self) -> float:
        return min(1.0, self.climb_rate / self.cruise_speed)

    def distance_at_time(self, elapsed: float) -> float:
        t = max(0.0, elapsed)
        ramp = self.cruise_speed / self.longitudinal_accel
        return (0.5 * self.longitudinal_accel * t * t if t < ramp
                else self.cruise_speed * (t - 0.5 * ramp))

    def speed_at_time(self, elapsed: float) -> float:
        return min(self.cruise_speed, self.longitudinal_accel * max(0.0, elapsed))


def map_scale(shape: tuple[int, ...], voxel_size: float) -> np.ndarray:
    if len(shape) != 3 or min(shape) < 2 or not math.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("性能约束需要有效三维体素地图和正 voxel_size")
    return np.asarray((shape[0] - 1, shape[2] - 1, shape[1] - 1), dtype=float) * voxel_size / 100


def _points(path: list[MapPoint]) -> np.ndarray:
    p = np.asarray(path, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all() or (p < 0).any() or (p > 100).any():
        raise ValueError("路径需要至少两个 [0,100] 内的有限三维点")
    if (np.linalg.norm(np.diff(p, axis=0), axis=1) < 1e-10).any():
        raise ValueError("路径包含重复点")
    return p


def inflate_obstacles(occupancy: np.ndarray, radius: float, voxel_size: float) -> np.ndarray:
    """Conservative Euclidean dilation, covering the closed source voxel cubes.

    A truncated, separable squared-distance transform uses NumPy only. The
    half-voxel diagonal accounts for cube surfaces rather than point obstacles.
    The original occupancy is never changed, and endpoints are never cleared.
    """
    occ = np.asarray(occupancy)
    map_scale(occ.shape, voxel_size)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("膨胀半径无效")
    if radius == 0 or not occ.any():
        return (occ != 0).astype(np.uint8)
    reach = radius / voxel_size + math.sqrt(3) / 2
    dist = np.where(occ != 0, 0.0, np.inf)
    for axis, n in enumerate(occ.shape):
        source = dist
        target = source.copy()
        for offset in range(1, min(n, int(math.ceil(reach)) + 1)):
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = slice(0, n - offset), slice(offset, n)
            lo, hi = tuple(lo), tuple(hi)
            np.minimum(target[lo], source[hi] + offset * offset, out=target[lo])
            np.minimum(target[hi], source[lo] + offset * offset, out=target[hi])
        dist = target
    return (dist <= reach * reach + 1e-9).astype(np.uint8)


def segment_free(a: np.ndarray, b: np.ndarray, occupancy: np.ndarray,
                 voxel_size: float, margin: float = 0.0) -> bool:
    """Exact segment/closed-voxel intersection; input axes are physical XYZ."""
    a, b = np.asarray(a)[[0, 2, 1]] / voxel_size, np.asarray(b)[[0, 2, 1]] / voxel_size
    bounds = np.asarray(occupancy.shape) - 1
    if (np.minimum(a, b) < -1e-8).any() or (np.maximum(a, b) > bounds + 1e-8).any():
        return False
    half = 0.5 + margin / voxel_size
    lo = np.maximum(0, np.ceil(np.minimum(a, b) - half).astype(int))
    hi = np.minimum(bounds, np.floor(np.maximum(a, b) + half).astype(int))
    region = occupancy[tuple(slice(int(x), int(y) + 1) for x, y in zip(lo, hi))]
    occupied = np.argwhere(region != 0) + lo
    if len(occupied) == 0:
        return True
    entry, leave = np.zeros(len(occupied)), np.ones(len(occupied))
    for axis, delta in enumerate(b - a):
        lower, upper = occupied[:, axis] - half, occupied[:, axis] + half
        if abs(delta) < 1e-12:
            outside = (a[axis] < lower) | (a[axis] > upper)
            entry[outside], leave[outside] = 1, 0
        else:
            t1, t2 = (lower - a[axis]) / delta, (upper - a[axis]) / delta
            entry = np.maximum(entry, np.minimum(t1, t2))
            leave = np.minimum(leave, np.maximum(t1, t2))
    return not bool((entry <= leave + 1e-12).any())


@lru_cache(maxsize=256)
def _supercover(delta: GridPoint) -> tuple[GridPoint, ...]:
    """All closed voxel cubes intersected by an integer-grid move."""
    d = np.asarray(delta, dtype=float)
    offsets = []
    for x in range(min(0, delta[0]), max(0, delta[0]) + 1):
        for y in range(min(0, delta[1]), max(0, delta[1]) + 1):
            for z in range(min(0, delta[2]), max(0, delta[2]) + 1):
                node = (x, y, z)
                if node == (0, 0, 0):
                    continue
                entry, leave = 0.0, 1.0
                for c, step in zip(node, d):
                    if step == 0:
                        if abs(c) > 0.5:
                            leave = -1
                    else:
                        p, q = (c - 0.5) / step, (c + 0.5) / step
                        entry, leave = max(entry, min(p, q)), min(leave, max(p, q))
                if entry <= leave + 1e-10:
                    offsets.append(node)
    return tuple(offsets)


def _shift(array: np.ndarray, delta: GridPoint, fill: float | bool) -> np.ndarray:
    result = np.full(array.shape, fill, dtype=array.dtype)
    src, dst = [], []
    for n, d in zip(array.shape, delta):
        if abs(d) >= n:
            return result
        src.append(slice(max(d, 0), min(n + d, n)))
        dst.append(slice(max(-d, 0), min(n - d, n)))
    result[tuple(dst)] = array[tuple(src)]
    return result


@dataclass
class FlightGrid:
    occupancy: np.ndarray
    actions: tuple[GridPoint, ...]
    valid_moves: np.ndarray
    costs: tuple[float, ...]
    voxel_size: float
    constraints: FlightConstraints


def prepare_flight_grid(occupancy: np.ndarray, voxel_size: float,
                        constraints: FlightConstraints) -> FlightGrid:
    constraints.validate()
    occ = inflate_obstacles(occupancy, constraints.turn_radius, voxel_size)
    fraction = constraints.vertical_fraction
    candidates = {(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1)
                  for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)}
    # Sub-voxel climb slopes need longer horizontal moves; otherwise a 26-only
    # graph would incorrectly make every ascent unreachable at low climb rates.
    if fraction < 1:
        run = math.sqrt(max(0, 1 / fraction ** 2 - 1))
        for x, z in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            k = max(1, int(math.ceil(run / math.hypot(x, z) - 1e-10)))
            if abs(x * k) < occ.shape[0] and abs(z * k) < occ.shape[2]:
                candidates.update((x * k, y, z * k) for y in (-1, 1))
    actions = tuple(sorted(a for a in candidates if abs(a[1]) / math.sqrt(sum(v*v for v in a)) <= fraction + 1e-10))
    free = occ == 0
    masks = []
    for action in actions:
        valid = free.copy()
        for offset in _supercover(action):
            valid &= _shift(free, offset, False)
        masks.append(valid)
    return FlightGrid(occ, actions, np.stack(masks),
                      tuple(voxel_size * math.sqrt(sum(v*v for v in a)) for a in actions),
                      voxel_size, constraints)


def grid_index(point: MapPoint, shape: tuple[int, ...]) -> GridPoint:
    p = np.asarray(point, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or (p > 100).any():
        raise ValueError("起终点必须是 [0,100] 内的有限三维坐标")
    return tuple(int(round(v)) for v in p[[0, 2, 1]] * (np.asarray(shape) - 1) / 100)


def grid_to_map(node: GridPoint, shape: tuple[int, ...]) -> MapPoint:
    p = np.asarray(node, dtype=float) / (np.asarray(shape) - 1) * 100
    return tuple(float(v) for v in p[[0, 2, 1]])


def turn_cost(previous: np.ndarray | None, direction: np.ndarray, since_turn: float,
              constraints: FlightConstraints) -> float:
    if previous is None:
        return 0.0
    cosine = float(np.dot(previous, direction) / (np.linalg.norm(previous) * np.linalg.norm(direction)))
    bend = max(0.0, 1 - min(1.0, cosine))
    proximity = max(0.0, 1 - since_turn / constraints.turn_radius)
    return bend * (constraints.turn_penalty + constraints.close_turn_penalty * proximity)


def _direction_changed(previous: np.ndarray | None, direction: np.ndarray) -> bool:
    return previous is not None and np.dot(previous, direction) / (np.linalg.norm(previous) * np.linalg.norm(direction)) < 1 - 1e-8


def _grid_search(grid: FlightGrid, start: GridPoint, goal: GridPoint,
                 progress_callback: Callable[[int], None] | None = None,
                 full_distance: bool = False) -> tuple[list[GridPoint], np.ndarray]:
    """Ordinary distance A*, preferring fewer/less closely spaced turns on ties.

    Reverse Dijkstra (full_distance) uses exactly the same feasible move graph.
    Turn context is a tie breaker, not a claim of optimal heading-state search.
    """
    shape = grid.occupancy.shape
    distance = np.full(shape, np.inf)
    if grid.occupancy[start] or grid.occupancy[goal]:
        return [], distance
    if full_distance:
        # Flat indices avoid retaining a heading/parent dictionary for every
        # voxel in a full-map teacher. Moves already include boundary checks.
        flat = distance.ravel()
        valid = grid.valid_moves.reshape(len(grid.actions), -1)
        offsets = [x * shape[1] * shape[2] + y * shape[2] + z for x, y, z in grid.actions]
        origin = int(np.ravel_multi_index(start, shape))
        flat[origin] = 0.0
        queue = [(0.0, origin)]
        popped, free_count = 0, max(1, int((grid.occupancy == 0).sum()))
        while queue:
            cost, cur = heapq.heappop(queue)
            if cost > flat[cur] + 1e-9:
                continue
            popped += 1
            for a in np.flatnonzero(valid[:, cur]):
                nxt, cand = cur + offsets[a], cost + grid.costs[a]
                if cand < flat[nxt] - 1e-9:
                    flat[nxt] = cand
                    heapq.heappush(queue, (cand, nxt))
            if progress_callback is not None and popped % 2048 == 0:
                progress_callback(min(80, int(popped * 80 / free_count)))
        return [], distance
    distance[start] = 0
    quality = {start: 0.0}
    parents: dict[GridPoint, GridPoint] = {}
    context: dict[GridPoint, tuple[np.ndarray | None, float]] = {start: (None, math.inf)}
    heuristic = lambda n: grid.voxel_size * math.dist(n, goal)
    heap = [(heuristic(start), 0.0, 0.0, start)]
    popped = 0
    while heap:
        _, penalty, cost, cur = heapq.heappop(heap)
        if cost > distance[cur] + 1e-9 or penalty > quality[cur] + 1e-9:
            continue
        popped += 1
        if cur == goal:
            break
        previous, since = context[cur]
        for a in np.flatnonzero(grid.valid_moves[(slice(None),) + cur]):
            delta, step = grid.actions[a], grid.costs[a]
            nxt = tuple(c + d for c, d in zip(cur, delta))
            cand = cost + step
            direction = np.asarray(delta, dtype=float)
            extra = turn_cost(previous, direction, since, grid.constraints)
            soft = penalty + extra
            if cand > distance[nxt] + 1e-9 or (abs(cand - distance[nxt]) <= 1e-9 and soft >= quality.get(nxt, math.inf) - 1e-9):
                continue
            distance[nxt], quality[nxt], parents[nxt] = cand, soft, cur
            context[nxt] = (direction, step if _direction_changed(previous, direction) else since + step)
            heapq.heappush(heap, (cand + heuristic(nxt), soft, cand, nxt))
    if not math.isfinite(distance[goal]):
        return [], distance
    path = [goal]
    while path[-1] != start:
        path.append(parents[path[-1]])
    return list(reversed(path)), distance


@dataclass(frozen=True)
class TrajectorySegment:
    start: MapPoint
    end: MapPoint
    length: float
    center: MapPoint | None = None
    tangent: MapPoint | None = None
    angle: float = 0.0
    radius: float = 0.0

    def position(self, distance: float) -> np.ndarray:
        t = min(1.0, max(0.0, distance / self.length))
        a, b = np.asarray(self.start), np.asarray(self.end)
        if self.center is None:
            return a + t * (b - a)
        c = np.asarray(self.center)
        return c + (a - c) * math.cos(t * self.angle) + np.asarray(self.tangent) * self.radius * math.sin(t * self.angle)


@dataclass
class FlightTrajectory:
    segments: list[TrajectorySegment]
    scale: np.ndarray
    polyline_map: list[MapPoint]
    constraints: FlightConstraints

    @property
    def length(self) -> float:
        return sum(s.length for s in self.segments)

    def position_map(self, distance: float) -> MapPoint:
        for segment in self.segments:
            if distance <= segment.length:
                return tuple(float(v) for v in segment.position(max(0, distance)) / self.scale)
            distance -= segment.length
        return tuple(float(v) for v in np.asarray(self.segments[-1].end) / self.scale)

    def samples_map(self) -> list[MapPoint]:
        # Rendering samples are independent of exact distance/time integration.
        result = [tuple(float(v) for v in np.asarray(self.segments[0].start) / self.scale)]
        for s in self.segments:
            count = max(1, int(math.ceil(s.angle / math.radians(2)))) if s.center is not None else 1
            result.extend(tuple(float(v) for v in s.position(s.length * i / count) / self.scale)
                          for i in range(1, count + 1))
        return result


def _climb_ok(a: np.ndarray, b: np.ndarray, config: FlightConstraints) -> bool:
    length = np.linalg.norm(b - a)
    return length > 1e-10 and abs(b[2] - a[2]) / length <= config.vertical_fraction + 1e-10


def _simplify(points: np.ndarray, occ: np.ndarray, size: float, config: FlightConstraints,
              shortcuts: bool) -> np.ndarray:
    kept = [points[0]]
    for i in range(1, len(points) - 1):
        u, v = points[i] - kept[-1], points[i+1] - points[i]
        if np.linalg.norm(np.cross(u, v)) > 1e-8 or np.dot(u, v) <= 0:
            kept.append(points[i])
    kept.append(points[-1])
    points = np.asarray(kept)
    if not shortcuts:
        return points
    kept, i = [points[0]], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not (_climb_ok(points[i], points[j], config) and segment_free(points[i], points[j], occ, size)):
            j -= 1
        kept.append(points[j])
        i = j
    return np.asarray(kept)


def _arc_segments(points: np.ndarray, config: FlightConstraints) -> list[TrajectorySegment]:
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    dirs = vectors / lengths[:, None]
    trims = np.zeros(len(points))
    angles = np.zeros(len(points))
    for i in range(1, len(points) - 1):
        angle = math.acos(float(np.clip(np.dot(dirs[i-1], dirs[i]), -1, 1)))
        if angle > math.pi - 1e-7:
            raise ValueError("路径包含掉头，无法按巡航转弯半径构造圆弧")
        if angle > 1e-7:
            angles[i], trims[i] = angle, config.turn_radius * math.tan(angle / 2)
    if (trims[:-1] + trims[1:] > lengths + 1e-8).any():
        raise ValueError("相邻转角过近或端点距转角过近，无法容纳规定半径圆弧；请降低巡航速度或调整起终点")
    segments = []
    def straight(a, b):
        length = float(np.linalg.norm(b - a))
        if length > 1e-8:
            segments.append(TrajectorySegment(tuple(a), tuple(b), length))
    cursor = points[0]
    for i in range(1, len(points) - 1):
        if angles[i] == 0:
            continue
        u, v, theta = dirs[i-1], dirs[i], angles[i]
        entry, exit = points[i] - u * trims[i], points[i] + v * trims[i]
        q = (v - math.cos(theta) * u) / math.sin(theta)
        center = entry + q * config.turn_radius
        # Exact maximum vertical tangent component, including interior extrema.
        extrema = [0.0, theta]
        root = math.atan2(q[2], u[2])
        extrema.extend(root + k * math.pi for k in range(-2, 3) if 0 <= root + k * math.pi <= theta)
        peak = max(abs(u[2] * math.cos(t) + q[2] * math.sin(t)) for t in extrema)
        if peak > config.vertical_fraction + 1e-9:
            raise ValueError("圆弧内爬升/下降速率超过飞机性能")
        straight(cursor, entry)
        segments.append(TrajectorySegment(tuple(entry), tuple(exit), config.turn_radius * theta,
                                          tuple(center), tuple(u), theta, config.turn_radius))
        cursor = exit
    straight(cursor, points[-1])
    return segments


def _check_segments(segments: list[TrajectorySegment], occupancy: np.ndarray,
                    size: float, config: FlightConstraints) -> None:
    for s in segments:
        if s.center is None:
            if not _climb_ok(np.asarray(s.start), np.asarray(s.end), config):
                raise ValueError("直线段爬升/下降速率超过飞机性能")
            count, margin = 1, 0.0
        else:
            # Inflate obstacles by the exact chord sagitta to conservatively
            # certify the continuous arc, not just sampled points/chords.
            step = min(math.radians(2), 2 * math.acos(max(-1.0, 1 - size / (100 * s.radius))))
            count = max(1, int(math.ceil(s.angle / step)))
            margin = s.radius * (1 - math.cos(s.angle / count / 2))
            u, q = np.asarray(s.tangent), (np.asarray(s.center) - np.asarray(s.start)) / s.radius
            # Arc coordinate extrema ensure it stays inside the finite map.
            extrema = [0.0, s.angle]
            for axis in range(3):
                root = math.atan2(-u[axis], q[axis])
                extrema.extend(root + k * math.pi for k in range(-2, 3) if 0 <= root + k * math.pi <= s.angle)
            upper = map_scale(occupancy.shape, size) * 100
            for t in extrema:
                p = s.position(t / s.angle * s.length)
                if (p < -1e-8).any() or (p > upper + 1e-8).any():
                    raise ValueError("转弯圆弧超出地图边界")
        for i in range(count):
            if not segment_free(s.position(s.length * i / count), s.position(s.length * (i+1) / count), occupancy, size, margin):
                raise ValueError("圆弧或直线轨迹与地形/障碍物相交")


def build_flight_trajectory(path: list[MapPoint], occupancy: np.ndarray, voxel_size: float,
                            constraints: FlightConstraints, grid: FlightGrid | None = None) -> FlightTrajectory:
    constraints.validate()
    scale = map_scale(occupancy.shape, voxel_size)
    physical = _points(path) * scale
    grid = grid or prepare_flight_grid(occupancy, voxel_size, constraints)
    error = "无法构造性能约束轨迹"
    for shortcuts in (True, False):
        points = _simplify(physical, grid.occupancy, voxel_size, constraints, shortcuts)
        try:
            segments = _arc_segments(points, constraints)
            _check_segments(segments, occupancy, voxel_size, constraints)
            return FlightTrajectory(segments, scale, [tuple(float(v) for v in p / scale) for p in points], constraints)
        except ValueError as exc:
            error = str(exc)
    raise ValueError(error)


def plan_flight_astar(start_xyz: MapPoint, end_xyz: MapPoint, occupancy: np.ndarray,
                      voxel_size: float, constraints: FlightConstraints) -> FlightTrajectory:
    grid = prepare_flight_grid(occupancy, voxel_size, constraints)
    start, goal = grid_index(start_xyz, grid.occupancy.shape), grid_index(end_xyz, grid.occupancy.shape)
    if grid.occupancy[start] or grid.occupancy[goal]:
        raise ValueError("起点或终点落在按转弯半径膨胀的障碍中；请调整位置或性能参数")
    if start == goal and math.dist(start_xyz, end_xyz) > 1e-8:
        return build_flight_trajectory([start_xyz, end_xyz], occupancy, voxel_size, constraints, grid)
    nodes, _ = _grid_search(grid, start, goal)
    if len(nodes) < 2:
        raise ValueError("A* 在膨胀地图和爬升限制下未找到可通行路径")
    path = [grid_to_map(n, grid.occupancy.shape) for n in nodes]
    if math.dist(start_xyz, path[0]) > 1e-8:
        path.insert(0, start_xyz)
    if math.dist(end_xyz, path[-1]) > 1e-8:
        path.append(end_xyz)
    return build_flight_trajectory(path, occupancy, voxel_size, constraints, grid)


def build_flight_teacher(occupancy: np.ndarray, end_xyz: MapPoint, voxel_size: float,
                         constraints: FlightConstraints, gamma: float, alpha: float,
                         progress_callback: Callable[[int], None] | None) -> dict[str, Any]:
    """Full-map targets on the same inflated, climb-limited action graph."""
    grid = prepare_flight_grid(occupancy, voxel_size, constraints)
    goal = grid_index(end_xyz, grid.occupancy.shape)
    if grid.occupancy[goal]:
        raise ValueError("终点位于按转弯半径膨胀的障碍中")
    _, dist = _grid_search(grid, goal, goal, progress_callback, full_distance=True)
    values = np.full(dist.shape, -1e9)
    finite = np.isfinite(dist)
    discounted = gamma ** (dist[finite] / voxel_size)
    values[finite] = 260 * discounted - (1 - discounted) / max(1e-9, 1 - gamma)
    q = np.full(dist.shape + (len(grid.actions),), -1e9, dtype=np.float32)
    for a, (delta, cost) in enumerate(zip(grid.actions, grid.costs)):
        future = _shift(values, delta, -1e9)
        is_goal = np.zeros(dist.shape, dtype=bool)
        is_goal[goal] = True
        reward = np.where(_shift(is_goal, delta, False), 260.0, -cost / voxel_size)
        q[..., a] = np.where(grid.valid_moves[a], reward + alpha * gamma * future, -1e9)
    return dict(q_table=q, policy=np.argmax(q, axis=3).astype(np.int8),
                free_mask=grid.occupancy == 0, goal_idx=goal, grid_shape=dist.shape,
                is_3d=True, actions=grid.actions, distance_map=dist,
                flight_constraints=constraints, flight_voxel_size=voxel_size,
                flight_valid_moves=grid.valid_moves)


def build_flight_policy_path(model: Any, start_xyz: MapPoint, end_xyz: MapPoint,
                            action_scores: Callable[[GridPoint], np.ndarray] | None = None) -> list[MapPoint]:
    """History-aware action selection, with genuine turn/close-turn deductions.

    Neural inputs are retained; this penalty is a policy-extraction correction,
    not a claim that the existing position-only networks learn heading dynamics.
    Only feasible, unvisited, distance-descending actions are considered.
    """
    config = model.flight_constraints
    start = grid_index(start_xyz, model.grid_shape)
    goal = grid_index(end_xyz, model.grid_shape)
    if goal != model.goal_idx or not model.free_mask[start] or not np.isfinite(model.distance_map[start]):
        return []
    if start == goal and math.dist(start_xyz, end_xyz) > 1e-8:
        return [start_xyz, end_xyz]
    cur, path, previous, since = start, [start], None, math.inf
    visited = {start}
    while cur != goal:
        scores = np.asarray(action_scores(cur) if action_scores is not None else model.q_table[cur], dtype=float)
        best, best_score = None, -math.inf
        for a in np.flatnonzero(model.flight_valid_moves[(slice(None),) + cur]):
            delta = model.actions[a]
            nxt = tuple(c + d for c, d in zip(cur, delta))
            if nxt in visited or model.distance_map[nxt] >= model.distance_map[cur] - 1e-9:
                continue
            penalty = turn_cost(previous, np.asarray(delta), since, config)
            score = float(scores[a]) - penalty
            if math.isfinite(score) and score > best_score:
                best, best_score = (nxt, delta, penalty), score
        if best is None:
            return []
        cur, delta, penalty = best
        step = model.flight_voxel_size * math.sqrt(sum(v*v for v in delta))
        since = step if _direction_changed(previous, np.asarray(delta)) else since + step
        previous = np.asarray(delta, dtype=float)
        path.append(cur)
        visited.add(cur)
    if len(path) < 2:
        return []
    result = [grid_to_map(n, model.grid_shape) for n in path]
    if math.dist(start_xyz, result[0]) > 1e-8:
        result.insert(0, start_xyz)
    if math.dist(end_xyz, result[-1]) > 1e-8:
        result.append(end_xyz)
    return result


def copy_flight_context(model: Any, teacher: Any) -> None:
    """Keep the teacher's feasible graph attached to distilled models."""
    for field in ('flight_constraints', 'flight_voxel_size', 'flight_valid_moves'):
        setattr(model, field, getattr(teacher, field))
