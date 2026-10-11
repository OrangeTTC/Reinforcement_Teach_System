"""Speed-state/action teaching heads and acceleration-limited time profiles.

Creator: Orange JYJ
License: Apache-2.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

ACCELERATION_ACTIONS = (-1.0, 0.0, 1.0)


def speed_features(speed: float, limit: float, next_limit: float, distance: float,
                   remaining: float, config: Any) -> np.ndarray:
    """Current speed, local/lookahead speed limits, action scale and goal distance."""
    return np.asarray([speed / config.max_speed, limit / config.max_speed,
                       next_limit / config.max_speed,
                       min(1.0, 2 * config.longitudinal_accel * distance / config.max_speed**2),
                       remaining / (remaining + config.max_speed)], dtype=np.float32)


def _speed_targets(features: np.ndarray) -> np.ndarray:
    """One-step model-based action values with travel-time and safety costs.

    This mirrors the existing full-map teacher/distillation teaching approach;
    these are not claimed to be a new end-to-end model-free flight learner.
    """
    v, cap, ahead, step, remaining = features.T
    proposed = np.sqrt(np.maximum(0, v[:, None]**2 + step[:, None] * np.asarray(ACCELERATION_ACTIONS)))
    next_v = np.minimum(proposed, np.minimum(cap, ahead)[:, None])
    reward = -step[:, None] / np.maximum(.03, v[:, None] + next_v)
    # A speed-dependent bootstrap estimates the remaining traversal cost.
    future = -remaining[:, None] / np.maximum(.08, next_v)
    excess = np.maximum(0, proposed - np.minimum(cap, ahead)[:, None])
    q = reward + .95 * future - 8 * excess
    q[(v < .005), :2] -= 10  # Holding/braking from rest cannot traverse an edge.
    return q.astype(np.float32)


@dataclass
class SpeedController:
    algorithm: str
    config: Any
    states: np.ndarray
    q_table: np.ndarray
    w1: np.ndarray | None = None
    b1: np.ndarray | None = None
    w2: np.ndarray | None = None
    b2: np.ndarray | None = None

    def scores(self, features: np.ndarray) -> np.ndarray:
        if self.w1 is None:
            # A sparse speed-state value table; state includes current velocity.
            index = int(np.argmin(np.sum((self.states - features)**2, axis=1)))
            return self.q_table[index]
        return np.tanh(features @ self.w1 + self.b1) @ self.w2 + self.b2

    def choose_action(self, speed: float, limit: float, next_limit: float,
                      distance: float, remaining: float) -> int:
        scores = self.scores(speed_features(speed, limit, next_limit, distance, remaining, self.config)).copy()
        if speed < 1e-8:
            scores[:2] = -np.inf
        return int(np.argmax(scores))


def train_speed_controller(algorithm: str, config: Any) -> SpeedController:
    """Build Q speed values / distil DQN, PPO and goal-conditioned HER heads.

    All four heads receive velocity and emit brake/hold/accelerate actions.
    HER additionally uses achieved-speed relabels in this teaching dataset.
    """
    config.validate()
    rng = np.random.default_rng(20261011)
    states = rng.uniform(0, 1, (2500, 5)).astype(np.float32)
    states[:, 1:3] = .1 + .9 * states[:, 1:3]
    states[:, 3] = .002 + .098 * states[:, 3]
    states[:100, 0] = 0
    if algorithm == 'HER':
        relabelled = states[:800].copy()
        relabelled[:, 2] = np.minimum(relabelled[:, 1], relabelled[:, 0] + .1)
        relabelled[:, 4] *= .5
        states = np.concatenate((states, relabelled))
    q = _speed_targets(states)
    head = SpeedController(algorithm, config, states, q)
    if algorithm == 'Q-learning':
        return head
    head.w1 = rng.normal(0, .2, (5, 24)).astype(np.float32)
    head.b1 = np.zeros(24, dtype=np.float32)
    head.w2 = rng.normal(0, .1, (24, 3)).astype(np.float32)
    head.b2 = np.zeros(3, dtype=np.float32)
    labels = np.argmax(q, axis=1)
    targets = q - q.mean(axis=1, keepdims=True)
    targets = np.clip(targets, -5, 5)
    for _ in range(220):
        rows = rng.integers(0, len(states), 256)
        x = states[rows]
        hidden = np.tanh(x @ head.w1 + head.b1)
        out = hidden @ head.w2 + head.b2
        if algorithm == 'DQN':
            grad = np.clip(out - targets[rows], -3, 3) / len(rows)
        else:
            prob = np.exp(out - out.max(axis=1, keepdims=True))
            prob /= prob.sum(axis=1, keepdims=True)
            prob[np.arange(len(rows)), labels[rows]] -= 1
            grad = prob / len(rows)
        dh = (grad @ head.w2.T) * (1 - hidden**2)
        head.w2 -= .035 * hidden.T @ grad
        head.b2 -= .035 * grad.sum(axis=0)
        head.w1 -= .035 * x.T @ dh
        head.b1 -= .035 * dh.sum(axis=0)
    return head


@dataclass
class SpeedProfile:
    distances: np.ndarray
    speeds: np.ndarray
    times: np.ndarray
    limits: np.ndarray
    actions: tuple[int, ...]

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def _state(self, elapsed: float) -> tuple[int, float, float]:
        t = float(np.clip(elapsed, 0, self.duration))
        i = min(len(self.times)-2, max(0, int(np.searchsorted(self.times, t, side='right'))-1))
        dt = t - self.times[i]
        acceleration = (self.speeds[i+1] - self.speeds[i]) / (self.times[i+1] - self.times[i])
        return i, dt, acceleration

    def distance_at_time(self, elapsed: float) -> float:
        i, dt, a = self._state(elapsed)
        return float(self.distances[i] + self.speeds[i] * dt + .5 * a * dt**2)

    def speed_at_time(self, elapsed: float) -> float:
        i, dt, a = self._state(elapsed)
        return float(self.speeds[i] + a * dt)


def segment_speed_limit(segment: Any, config: Any) -> float:
    """A conservative whole-segment cap, including exact vertical arc extrema."""
    cap = min(config.max_speed, config.cruise_speed)
    if segment.center is None:
        peak = abs(segment.end[2] - segment.start[2]) / segment.length
    else:
        cap = min(cap, math.sqrt(config.lateral_accel * segment.radius))
        u = np.asarray(segment.tangent)
        q = (np.asarray(segment.center)-np.asarray(segment.start)) / segment.radius
        root = math.atan2(q[2], u[2])
        angles = [0.0, segment.angle]
        angles.extend(root+k*math.pi for k in range(-2,3) if 0 <= root+k*math.pi <= segment.angle)
        peak = max(abs(u[2]*math.cos(t)+q[2]*math.sin(t)) for t in angles)
    if peak > 1e-12:
        cap = min(cap, config.climb_rate / peak)
    return cap


def build_speed_profile(trajectory: Any, controller: SpeedController | None = None) -> SpeedProfile:
    """Forward/backward reachable speeds and exact piecewise-acceleration timing.

    Both endpoints are at rest. Caps certify the entire geometric segment,
    and backward propagation begins braking before tighter curves or climbs.
    Learned speed actions act before the same safety projection for all heads.
    """
    config = trajectory.constraints
    distances, edge_caps = [0.0], []
    mesh = max(.1, trajectory.length / 4000)
    for segment in trajectory.segments:
        count = max(2, int(math.ceil(segment.length / mesh - 1e-10)))
        cap = segment_speed_limit(segment, config)
        base = distances[-1]
        distances.extend(base+segment.length*i/count for i in range(1,count+1))
        edge_caps.extend([cap]*count)
    distances, edge_caps = np.asarray(distances, dtype=float), np.asarray(edge_caps, dtype=float)
    ds = np.diff(distances)
    limits = np.concatenate(([edge_caps[0]], np.minimum(edge_caps[:-1],edge_caps[1:]), [edge_caps[-1]]))
    envelope = limits.copy()
    envelope[0] = envelope[-1] = 0
    a = config.longitudinal_accel
    for i in range(len(ds)):
        envelope[i+1] = min(envelope[i+1], math.sqrt(envelope[i]**2+2*a*ds[i]))
    for i in range(len(ds)-1,-1,-1):
        envelope[i] = min(envelope[i], math.sqrt(envelope[i+1]**2+2*a*ds[i]))
    speeds = envelope.copy()
    actions = []
    if controller is not None:
        for i, length in enumerate(ds):
            action = controller.choose_action(speeds[i], edge_caps[i], envelope[i+1],
                                              float(length), float(distances[-1]-distances[i]))
            next_speed = math.sqrt(max(0, speeds[i]**2+2*a*ACCELERATION_ACTIONS[action]*length))
            if i+1 < len(ds):
                # Prevent a zero-speed spatial edge; from rest accelerate.
                next_speed = max(next_speed, min(envelope[i+1], math.sqrt(a*length)))
            speeds[i+1] = min(envelope[i+1], next_speed)
            actions.append(action)
        for i in range(len(ds)-1,-1,-1):
            speeds[i] = min(speeds[i], math.sqrt(speeds[i+1]**2+2*a*ds[i]))
    else:
        actions = [2 if v1 > v0+1e-9 else 0 if v1 < v0-1e-9 else 1 for v0,v1 in zip(speeds[:-1],speeds[1:])]
    if np.any(speeds[:-1]+speeds[1:] <= 0):
        raise ValueError('速度策略未能生成可运行轨迹')
    times = np.concatenate(([0.0], np.cumsum(2*ds/(speeds[:-1]+speeds[1:]))))
    return SpeedProfile(distances,speeds,times,limits,tuple(actions))
