"""
Central registry for algorithm training/path entrypoints.

Creator: Orange JYJ
License: Apache-2.0
"""
from __future__ import annotations

"""Central registry of navigation algorithm entrypoints.

This file is intentionally lightweight and can be used as the single source
of truth for where each algorithm's train/build-path functions are defined.
"""

from typing import Callable

from algorithm.algorithm_dqn import build_dqn_path, train_dqn_full_map
from algorithm.algorithm_her import build_her_path, train_her_full_map
from algorithm.algorithm_ppo import build_ppo_path, train_ppo_full_map
from algorithm.algorithm_q_learning import build_q_learning_path, train_q_learning_full_map

# Teacher entry used by HER/DQN/PPO for supervised/full-map targets.
TEACHER_TRAIN_ENTRY = train_q_learning_full_map

# Public registry for all train entrypoints.
TRAIN_ENTRYPOINTS: dict[str, Callable[..., object]] = {
    "Q-learning": train_q_learning_full_map,
    "DQN": train_dqn_full_map,
    "PPO": train_ppo_full_map,
    "HER": train_her_full_map,
}

# Public registry for all path-building (navigation) entrypoints.
PATH_ENTRYPOINTS: dict[str, Callable[..., object]] = {
    "Q-learning": build_q_learning_path,
    "DQN": build_dqn_path,
    "PPO": build_ppo_path,
    "HER": build_her_path,
}


def list_entrypoint_names() -> dict[str, dict[str, str]]:
    """Return module-qualified names for display/logging."""

    def qname(func: Callable[..., object]) -> str:
        return f"{func.__module__}.{func.__name__}"

    return {
        "train": {k: qname(v) for k, v in TRAIN_ENTRYPOINTS.items()},
        "path": {k: qname(v) for k, v in PATH_ENTRYPOINTS.items()},
        "teacher": {"Q-learning": qname(TEACHER_TRAIN_ENTRY)},
    }
