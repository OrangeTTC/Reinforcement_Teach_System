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

from typing import Any, Callable

from algorithm.algorithm_dqn import build_dqn_path, train_dqn_full_map
from algorithm.algorithm_her import build_her_path, train_her_full_map
from algorithm.algorithm_ppo import build_ppo_path, train_ppo_full_map
from algorithm.algorithm_q_learning import build_q_learning_path, train_q_learning_full_map

# ---------------------------------------------------------------------------
# Normalized train signature used by all entries in TRAIN_ENTRYPOINTS:
#
#   train_xxx_full_map(
#       occupancy,
#       heights,
#       end_map_xyz: tuple[float, float, float] | None,
#       map_to_index: Callable[[float, int, str], int],
#       progress_callback: Callable[[int], None] | None = None,
#   ) -> tuple[bool, str, model_or_None]
#
# Q-learning's native signature returns extra statistics fields, so wrap it.
# ---------------------------------------------------------------------------


def _train_q_learning_normalized(
    occupancy: Any,
    heights: Any,
    end_map_xyz: tuple[float, float, float] | None,
    map_to_index: Callable[[float, int, str], int],
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[bool, str, Any]:
    """Normalized wrapper for Q-learning so it shares the standard train signature."""
    ok, model, episodes, free_cells, goal_idx = train_q_learning_full_map(
        occupancy=occupancy,
        heights=heights,
        end_map_xyz=end_map_xyz,
        map_to_index=map_to_index,
        progress_callback=progress_callback,
    )
    if not ok or model is None:
        return (False, "Q-learning 训练失败（无效地图或无法到达目标点）", None)
    gx, gy = goal_idx[0], goal_idx[1] if len(goal_idx) > 1 else (0, 0)
    msg = f"Q-learning 已训练：迭代={episodes}，自由格={free_cells}，目标=({gx},{gy})"
    return (True, msg, model)


# ---------------------------------------------------------------------------
# Normalized path-build signature used by all entries in PATH_ENTRYPOINTS:
#
#   build_xxx_path(
#       model,
#       start_xyz: tuple[float, float, float],
#       end_xyz:   tuple[float, float, float],
#       map_to_index: Callable[[float, int, str], int],
#   ) -> list[tuple[float, float, float]]
# ---------------------------------------------------------------------------

# Teacher entry used by HER/DQN/PPO for supervised/full-map targets.
TEACHER_TRAIN_ENTRY = train_q_learning_full_map

# Public registry for all train entrypoints.
# All callables must follow the normalized train signature above.
TRAIN_ENTRYPOINTS: dict[str, Callable[..., tuple[bool, str, Any]]] = {
    "Q-learning": _train_q_learning_normalized,
    "DQN": train_dqn_full_map,
    "PPO": train_ppo_full_map,
    "HER": train_her_full_map,
}

# Public registry for all path-building (navigation) entrypoints.
# All callables must follow the normalized path-build signature above.
PATH_ENTRYPOINTS: dict[str, Callable[..., list[tuple[float, float, float]]]] = {
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
