"""
Qt OpenGL widget for terrain rendering, editing, and navigation playback.

Creator: Orange JYJ
License: Apache-2.0
"""
from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QLabel

from algorithm.algorithm_a_star import plan_astar_path_map_xyz
from algorithm.algorithm_dqn import DQNModel, build_dqn_path, train_dqn_full_map
from algorithm.algorithm_her import HERModel, build_her_path, train_her_full_map
from algorithm.algorithm_ppo import PPOModel, build_ppo_path, train_ppo_full_map
from algorithm.algorithm_q_learning import QLearningModel, build_q_learning_path, train_q_learning_full_map
from map.map_terrain_edit import (
    derive_heightfield_from_occupancy,
    export_occupancy_csv,
    load_occupancy_csv,
    load_terrain_hdf5,
    save_occupancy_hdf5,
)


class RotatingCubeWidget(QOpenGLWidget):
    hover_text_changed = Signal(str)
    edit_state_changed = Signal(bool)

    def __init__(self, parent=None, terrain_path: str | None = None) -> None:
        super().__init__(parent)
        self._terrain_path = terrain_path
        self._heights = None
        self._display_list = None
        self._gl_ready = False

        self._cell_size = 1.0

        # Grid meta (for mapping world<->map coords)
        self._grid_rows = 0
        self._grid_cols = 0
        self._grid_ny = None  # vertical resolution, only known for occupancy-grid HDF5

        # Editable occupancy (Scheme 2). Stored as occ[x, height, y_plane] uint8.
        self._occupancy = None
        self._occ_nx = None
        self._occ_ny = None
        self._occ_nz = None
        self._undo_stack: list[Any] = []
        self._max_undo_steps = 20

        # World-space terrain extents/origin. X/Y are planar, Z is height.
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._origin_z = 0.0
        self._extent_x = 1.0
        self._extent_y = 1.0
        self._extent_z = None

        # Terrain stats used by the camera fit.
        self._terrain_half_x = 5.0
        self._terrain_half_y = 5.0
        self._terrain_hmax = 0.0

        # MATLAB-like camera controls
        self._yaw = 45.0
        self._pitch = -35.0
        self._distance = 12.0
        self._pan_x = 0.0
        self._pan_y = 0.0

        self._target_x = 0.0
        self._target_y = 0.0
        self._target_z = 0.0

        self._last_mouse_pos = None
        self._mouse_mode = None  # "orbit" | "pan" | None
        self._pending_mouse_mode = None  # "orbit" | "pan" | None

        # Click/drag disambiguation (so left-click can pin a coordinate without breaking orbit).
        self._press_pos = None
        self._press_button = None
        self._press_dragged = False
        self._click_drag_threshold_px = 8.0

        self._hover_pos = None

        # Pinned map coordinates (for tracing). Stored as list of (mx, my) in 0..100 map space.
        # If at least one pin exists, we display the most recently pinned coordinate.
        self._pinned_xy: list[tuple[float, float]] = []
        self._active_pin_index: int | None = None

        self._last_emitted_xyz_text: str | None = None

        # Cached hover pick in map coords (mx,my) and the last GL matrices for screen projection.
        self._last_hover_map_xy: tuple[float, float] | None = None
        self._last_gl_model: tuple[float, ...] | None = None
        self._last_gl_proj: tuple[float, ...] | None = None
        self._last_gl_viewport: tuple[int, int, int, int] | None = None
        self._last_gl_ratio: float = 1.0

        self._pin_screen_xy: list[tuple[float, float] | None] = []
        self._pin_labels: list[QLabel] = []

        # Trace points (start/end) and movable agent in map XYZ (0..100 coordinates).
        self._start_map_xyz: tuple[float, float, float] | None = None
        self._end_map_xyz: tuple[float, float, float] | None = None
        self._agent_map_xyz: tuple[float, float, float] | None = None
        self._agent_step_map = 2.0

        # A* navigation state.
        self._nav_speed_units_per_sec = 10.0
        self._nav_path_map_xyz: list[tuple[float, float, float]] = []
        self._nav_path_cumdist: list[float] = []
        self._nav_total_dist = 0.0
        self._nav_progress_dist = 0.0
        self._nav_running = False
        self._nav_last_time_s: float | None = None
        self._nav_timer = QTimer(self)
        self._nav_timer.setInterval(16)
        self._nav_timer.timeout.connect(self._on_nav_timer)

        # Q-learning navigation state.
        self._q_learning_trained = False
        self._q_learning_model: QLearningModel | None = None

        # Placeholders for other RL algorithms.
        self._dqn_trained = False
        self._dqn_model: DQNModel | None = None
        self._ppo_trained = False
        self._ppo_model: PPOModel | None = None
        self._her_trained = False
        self._her_model: HERModel | None = None

        self._start_label = QLabel(self)
        self._start_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._start_label.setStyleSheet("color: #7fffd4; background-color: rgba(0, 0, 0, 160); padding: 2px 4px;")
        self._start_label.hide()

        self._end_label = QLabel(self)
        self._end_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._end_label.setStyleSheet("color: #ffb366; background-color: rgba(0, 0, 0, 160); padding: 2px 4px;")
        self._end_label.hide()

        # Receive mouse move events even when no button is pressed.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self._show_axes = True

        self._emit_edit_state()


    def set_show_axes(self, show: bool) -> None:
        self._show_axes = bool(show)
        self.update()


    def tick(self) -> None:
        # No continuous animation.
        self.update()

    def can_undo_edit(self) -> bool:
        return bool(self._undo_stack) and self._occupancy is not None

    def undo_last_edit(self) -> bool:
        if not self.can_undo_edit():
            return False

        self._occupancy = self._undo_stack.pop()
        self._heights = self._derive_heightfield_from_occupancy(self._occupancy, self._cell_size)
        self._invalidate_q_learning()
        self._recompile_display_list_from_current_heights()
        self._emit_edit_state()
        return True

    def _emit_edit_state(self) -> None:
        self.edit_state_changed.emit(self.can_undo_edit())

    def _reset_edit_history(self) -> None:
        self._undo_stack.clear()
        self._emit_edit_state()

    def _push_undo_state(self) -> bool:
        if self._occupancy is None:
            self._emit_edit_state()
            return False

        self._undo_stack.append(self._occupancy.copy())
        if len(self._undo_stack) > self._max_undo_steps:
            self._undo_stack.pop(0)
        self._emit_edit_state()
        return True

    def initializeGL(self) -> None:  # noqa: N802 (Qt naming)
        from OpenGL import GL as gl

        gl.glClearColor(0.06, 0.06, 0.08, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glShadeModel(gl.GL_SMOOTH)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_LIGHTING)

        self._try_build_terrain_display_list()

        self._gl_ready = True

    def set_terrain_path(self, terrain_path: str | None) -> None:
        self._terrain_path = terrain_path
        if not self._gl_ready:
            return
        self._rebuild_terrain()

    def export_current_terrain_csv(self, output_path: str | Path) -> bool:
        """Export editable occupancy terrain to unified CSV format.

        Returns False when no editable occupancy terrain is loaded.
        """

        if self._occupancy is None:
            return False
        return export_occupancy_csv(self._occupancy, self._cell_size, output_path)

    def import_terrain_csv(self, csv_path: str | Path) -> bool:
        """Import terrain from unified CSV format into current editable terrain."""

        parsed = load_occupancy_csv(csv_path)
        if parsed is None:
            return False

        occ, voxel_size = parsed
        try:
            nx, ny, nz = occ.shape
            self._occupancy = occ
            self._occ_nx = int(nx)
            self._occ_ny = int(ny)
            self._occ_nz = int(nz)
            self._grid_ny = int(ny)
            self._cell_size = float(voxel_size)
            self._heights = self._derive_heightfield_from_occupancy(self._occupancy, self._cell_size)
            self._invalidate_q_learning()
            self._reset_edit_history()
            self._clear_pins()
            self._recompile_display_list_from_current_heights()
            self._emit_edit_state()
            return True
        except Exception:
            return False

    def save_current_terrain_hdf5(self, output_path: str | Path | None = None) -> bool:
        """Persist current editable occupancy terrain to an HDF5 file.

        Returns False when there is no editable occupancy terrain loaded.
        """

        if self._occupancy is None:
            return False

        target = Path(output_path) if output_path is not None else Path(self._terrain_path or "")
        if not str(target):
            return False

        return save_occupancy_hdf5(self._occupancy, self._cell_size, target)

    def _rebuild_terrain(self) -> None:
        """Rebuild the terrain display list in the current GL context."""

        from OpenGL import GL as gl

        self.makeCurrent()
        try:
            if self._display_list is not None:
                try:
                    gl.glDeleteLists(self._display_list, 1)
                except Exception:
                    pass
            self._display_list = None
            self._heights = None
            self._occupancy = None
            self._occ_nx = None
            self._occ_ny = None
            self._occ_nz = None
            self._invalidate_q_learning()
            self._reset_edit_history()
            self._clear_pins()

            self._try_build_terrain_display_list()
        finally:
            self.doneCurrent()

        self.update()

    def _emit_xyz_text(self, text: str) -> None:
        if text == self._last_emitted_xyz_text:
            return
        self._last_emitted_xyz_text = text
        self.hover_text_changed.emit(text)

    def _clear_pins(self) -> None:
        self.stop_navigation()
        self._pinned_xy.clear()
        self._active_pin_index = None
        self._last_hover_map_xy = None
        for lb in self._pin_labels:
            lb.hide()
            lb.deleteLater()
        self._pin_labels.clear()
        self._start_map_xyz = None
        self._end_map_xyz = None
        self._agent_map_xyz = None
        self._nav_path_map_xyz = []
        self._nav_path_cumdist = []
        self._nav_total_dist = 0.0
        self._nav_progress_dist = 0.0
        self._start_label.hide()
        self._end_label.hide()
        self._emit_xyz_text("XYZ: --")

    def clear_trace_points(self) -> None:
        self.stop_navigation()
        self._invalidate_q_learning()
        self._start_map_xyz = None
        self._end_map_xyz = None
        self._agent_map_xyz = None
        self._nav_path_map_xyz = []
        self._nav_path_cumdist = []
        self._nav_total_dist = 0.0
        self._nav_progress_dist = 0.0
        self._start_label.hide()
        self._end_label.hide()
        self.update()

    def _active_pin_map_xy(self) -> tuple[float, float] | None:
        if self._active_pin_index is None:
            return None
        if not (0 <= self._active_pin_index < len(self._pinned_xy)):
            return None
        return self._pinned_xy[self._active_pin_index]

    def _terrain_height_at_map(self, mx: float, my: float) -> float:
        """Return terrain height (world Z) at map coords mx,my in [0,100]."""

        heights = self._heights
        if heights is None:
            return 0.0

        rows, cols = heights.shape
        if rows < 2 or cols < 2:
            return float(heights[0, 0]) if rows >= 1 and cols >= 1 else 0.0

        mx = max(0.0, min(100.0, float(mx)))
        my = max(0.0, min(100.0, float(my)))

        x = mx / 100.0 * float(cols - 1)
        y = my / 100.0 * float(rows - 1)

        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = min(cols - 1, x0 + 1)
        y1 = min(rows - 1, y0 + 1)

        tx = x - float(x0)
        ty = y - float(y0)

        h00 = float(heights[y0, x0])
        h10 = float(heights[y0, x1])
        h01 = float(heights[y1, x0])
        h11 = float(heights[y1, x1])

        h0 = (1.0 - tx) * h00 + tx * h10
        h1 = (1.0 - tx) * h01 + tx * h11
        return (1.0 - ty) * h0 + ty * h1

    def _format_xyz_text_from_map(self, mx: float, my: float) -> str:
        wz = self._terrain_height_at_map(mx, my)
        if self._extent_z is not None and self._extent_z > 0:
            mz = wz / float(self._extent_z) * 100.0
            mz = max(0.0, min(100.0, mz))
            return f"XYZ: {mx:.1f}, {my:.1f}, {mz:.1f}"
        return f"XYZ: {mx:.1f}, {my:.1f}, {wz:.2f}"

    def _normalize_map_xyz(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        mx = max(0.0, min(100.0, float(x)))
        my = max(0.0, min(100.0, float(y)))
        mz = max(0.0, min(100.0, float(z)))
        return (mx, my, mz)

    def _map_xyz_equal(
        self,
        a: tuple[float, float, float] | None,
        b: tuple[float, float, float] | None,
        eps: float = 1e-6,
    ) -> bool:
        if a is None or b is None:
            return a is b
        return (
            abs(float(a[0]) - float(b[0])) <= eps
            and abs(float(a[1]) - float(b[1])) <= eps
            and abs(float(a[2]) - float(b[2])) <= eps
        )

    def _map_xyz_to_world(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        wx = self._origin_x + (float(x) / 100.0) * float(self._extent_x)
        wy = self._origin_y + (float(y) / 100.0) * float(self._extent_y)
        if self._extent_z is not None and self._extent_z > 0:
            wz = (float(z) / 100.0) * float(self._extent_z)
        else:
            wz = float(z)
        return (float(wx), float(wy), float(wz))

    def set_trace_points_map(
        self,
        sx: float,
        sy: float,
        sz: float,
        ex: float,
        ey: float,
        ez: float,
    ) -> bool:
        """Set trace start/end map coordinates (0..100) and spawn agent at start."""

        try:
            start = self._normalize_map_xyz(sx, sy, sz)
            end = self._normalize_map_xyz(ex, ey, ez)
        except Exception:
            return False

        end_changed = not self._map_xyz_equal(self._end_map_xyz, end)

        self._start_map_xyz = start
        self._end_map_xyz = end
        self._agent_map_xyz = start
        # Keep trained policy when only start changes; retraining is required only if end changes.
        if end_changed:
            self._invalidate_q_learning()
        self.stop_navigation()
        self._nav_path_map_xyz = []
        self._nav_path_cumdist = []
        self._nav_total_dist = 0.0
        self._nav_progress_dist = 0.0
        self.update()
        return True

    def move_agent_map(self, dx: float, dy: float, dz: float) -> bool:
        """Move agent by map-space delta (0..100 coordinates)."""

        if self._agent_map_xyz is None:
            return False

        ax, ay, az = self._agent_map_xyz
        nx, ny, nz = self._normalize_map_xyz(ax + float(dx), ay + float(dy), az + float(dz))
        self._agent_map_xyz = (nx, ny, nz)
        self.stop_navigation()
        self._nav_path_map_xyz = []
        self._nav_path_cumdist = []
        self._nav_total_dist = 0.0
        self._nav_progress_dist = 0.0
        self.update()
        return True

    def reset_agent_to_start(self) -> bool:
        if self._start_map_xyz is None:
            return False
        self._agent_map_xyz = self._start_map_xyz
        self.stop_navigation()
        self._nav_path_map_xyz = []
        self._nav_path_cumdist = []
        self._nav_total_dist = 0.0
        self._nav_progress_dist = 0.0
        self.update()
        return True

    def set_navigation_speed_units_per_sec(self, speed: float) -> None:
        try:
            s = float(speed)
        except Exception:
            s = 10.0
        self._nav_speed_units_per_sec = max(5.0, min(20.0, s))

    def get_last_navigation_length(self) -> float:
        return float(self._nav_total_dist)

    def get_last_navigation_path(self) -> list[tuple[float, float, float]]:
        """Return a copy of the latest planned navigation path in map XYZ."""

        return list(self._nav_path_map_xyz)

    def replay_navigation_path(self, path: list[tuple[float, float, float]]) -> bool:
        """Replay a previously recorded map-space path."""

        if not path or len(path) < 2:
            return False
        # Copy path to avoid accidental external mutation during playback.
        return self._set_navigation_path(list(path))

    def _path_length(self, planned: list[tuple[float, float, float]]) -> float:
        if not planned or len(planned) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(planned)):
            x0, y0, z0 = planned[i - 1]
            x1, y1, z1 = planned[i]
            total += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        return total

    def _line_of_sight_free_map(
        self,
        a: tuple[float, float, float],
        b: tuple[float, float, float],
    ) -> bool:
        """Check straight-line traversability in map-space for occupancy terrain."""

        if self._occupancy is None or self._occ_nx is None or self._occ_ny is None or self._occ_nz is None:
            return True

        nx, ny_h, nz = int(self._occ_nx), int(self._occ_ny), int(self._occ_nz)

        ax = self._map_to_index(a[0], nx, "round")
        ay_h = self._map_to_index(a[2], ny_h, "round")
        az = self._map_to_index(a[1], nz, "round")

        bx = self._map_to_index(b[0], nx, "round")
        by_h = self._map_to_index(b[2], ny_h, "round")
        bz = self._map_to_index(b[1], nz, "round")

        if bool(self._occupancy[ax, ay_h, az]) or bool(self._occupancy[bx, by_h, bz]):
            return False

        dx = bx - ax
        dy_h = by_h - ay_h
        dz = bz - az
        steps = max(abs(dx), abs(dy_h), abs(dz))
        if steps <= 0:
            return True

        # Oversample slightly to reduce corner miss-through on coarse grids.
        samples = max(2, steps * 2)
        for s in range(samples + 1):
            t = float(s) / float(samples)
            ix = int(round(ax + dx * t))
            iy_h = int(round(ay_h + dy_h * t))
            iz = int(round(az + dz * t))
            ix = max(0, min(nx - 1, ix))
            iy_h = max(0, min(ny_h - 1, iy_h))
            iz = max(0, min(nz - 1, iz))
            if bool(self._occupancy[ix, iy_h, iz]):
                return False
        return True

    def _optimize_rl_path(self, planned: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        """Prune unnecessary RL waypoints while preserving obstacle safety."""

        if not planned or len(planned) < 3:
            return planned

        pruned = [planned[0]]
        i = 0
        n = len(planned)
        while i < n - 1:
            j = n - 1
            best = i + 1
            while j > i + 1:
                if self._line_of_sight_free_map(planned[i], planned[j]):
                    best = j
                    break
                j -= 1
            pruned.append(planned[best])
            i = best

        # Keep optimization only if it truly shortens trajectory.
        return pruned if self._path_length(pruned) + 1e-6 < self._path_length(planned) else planned

    def _invalidate_q_learning(self) -> None:
        self._q_learning_trained = False
        self._q_learning_model = None
        self._dqn_trained = False
        self._dqn_model = None
        self._ppo_trained = False
        self._ppo_model = None
        self._her_trained = False
        self._her_model = None

    def is_algorithm_trained(self, algorithm: str) -> bool:
        algo = str(algorithm).strip()
        if algo == "A*":
            return True
        if algo == "Q-learning":
            return bool(self._q_learning_trained)
        if algo == "DQN":
            return bool(self._dqn_trained)
        if algo == "PPO":
            return bool(self._ppo_trained)
        if algo == "HER":
            return bool(self._her_trained)
        return False

    def train_navigation_policy(
        self,
        algorithm: str,
        progress_callback: Callable[[int], None] | None = None,
    ) -> tuple[bool, str]:
        algo = str(algorithm).strip()
        if algo == "A*":
            return (True, "A* does not require training")
        if algo == "Q-learning":
            if self._end_map_xyz is None:
                self._invalidate_q_learning()
                return (False, "Set End point before Q-learning training")

            trained, model, episodes, free_cells, goal_idx = self._train_q_learning_full_map(progress_callback)
            if not trained:
                self._invalidate_q_learning()
                return (False, "Q-learning training failed (invalid map or unreachable goal)")

            self._q_learning_model = model
            self._q_learning_trained = True

            gx, gy = goal_idx
            return (True, f"Q-learning trained: episodes={episodes}, free={free_cells}, goal=({gx},{gy})")

        if algo == "DQN":
            ok, msg, model = train_dqn_full_map(
                occupancy=self._occupancy,
                heights=self._heights,
                end_map_xyz=self._end_map_xyz,
                map_to_index=self._map_to_index,
                progress_callback=progress_callback,
            )
            self._dqn_trained = bool(ok)
            self._dqn_model = model if ok else None
            return (ok, msg)

        if algo == "PPO":
            ok, msg, model = train_ppo_full_map(
                occupancy=self._occupancy,
                heights=self._heights,
                end_map_xyz=self._end_map_xyz,
                map_to_index=self._map_to_index,
                progress_callback=progress_callback,
            )
            self._ppo_trained = bool(ok)
            self._ppo_model = model if ok else None
            return (ok, msg)

        if algo == "HER":
            ok, msg, model = train_her_full_map(
                occupancy=self._occupancy,
                heights=self._heights,
                end_map_xyz=self._end_map_xyz,
                map_to_index=self._map_to_index,
                progress_callback=progress_callback,
            )
            self._her_trained = bool(ok)
            self._her_model = model if ok else None
            return (ok, msg)

        return (False, f"{algo} is unknown")

    def _set_navigation_path(self, planned: list[tuple[float, float, float]]) -> bool:
        if not planned or len(planned) < 2:
            return False

        self._nav_path_map_xyz = planned
        self._nav_path_cumdist = [0.0]
        acc = 0.0
        for i in range(1, len(planned)):
            x0, y0, z0 = planned[i - 1]
            x1, y1, z1 = planned[i]
            acc += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
            self._nav_path_cumdist.append(acc)
        self._nav_total_dist = acc
        self._nav_progress_dist = 0.0
        self._agent_map_xyz = planned[0]

        self._nav_running = True
        self._nav_last_time_s = time.perf_counter()
        if not self._nav_timer.isActive():
            self._nav_timer.start()
        self.update()
        return True

    def _train_q_learning_full_map(
        self,
        progress_callback: Callable[[int], None] | None = None,
    ) -> tuple[bool, QLearningModel | None, int, int, tuple[int, int]]:
        return train_q_learning_full_map(
            occupancy=self._occupancy,
            heights=self._heights,
            end_map_xyz=self._end_map_xyz,
            map_to_index=self._map_to_index,
            progress_callback=progress_callback,
        )

    def start_q_learning_navigation(self) -> bool:
        if not self._q_learning_trained or self._q_learning_model is None:
            return False
        if self._end_map_xyz is None:
            return False

        start_xyz = self._agent_map_xyz if self._agent_map_xyz is not None else self._start_map_xyz
        if start_xyz is None:
            return False

        planned = build_q_learning_path(
            model=self._q_learning_model,
            start_xyz=start_xyz,
            end_xyz=self._end_map_xyz,
            map_to_index=self._map_to_index,
        )
        if not planned:
            return False
        planned = self._optimize_rl_path(planned)
        return self._set_navigation_path(planned)

    def start_dqn_navigation(self) -> bool:
        if not self._dqn_trained or self._dqn_model is None:
            return False
        if self._end_map_xyz is None:
            return False

        start_xyz = self._agent_map_xyz if self._agent_map_xyz is not None else self._start_map_xyz
        if start_xyz is None:
            return False

        planned = build_dqn_path(
            model=self._dqn_model,
            start_xyz=start_xyz,
            end_xyz=self._end_map_xyz,
            map_to_index=self._map_to_index,
        )
        if not planned:
            return False
        planned = self._optimize_rl_path(planned)
        return self._set_navigation_path(planned)

    def start_ppo_navigation(self) -> bool:
        if not self._ppo_trained or self._ppo_model is None:
            return False
        if self._end_map_xyz is None:
            return False

        start_xyz = self._agent_map_xyz if self._agent_map_xyz is not None else self._start_map_xyz
        if start_xyz is None:
            return False

        planned = build_ppo_path(
            model=self._ppo_model,
            start_xyz=start_xyz,
            end_xyz=self._end_map_xyz,
            map_to_index=self._map_to_index,
        )
        if not planned:
            return False
        planned = self._optimize_rl_path(planned)
        return self._set_navigation_path(planned)

    def start_her_navigation(self) -> bool:
        if not self._her_trained or self._her_model is None:
            return False
        if self._end_map_xyz is None:
            return False

        start_xyz = self._agent_map_xyz if self._agent_map_xyz is not None else self._start_map_xyz
        if start_xyz is None:
            return False

        planned = build_her_path(
            model=self._her_model,
            start_xyz=start_xyz,
            end_xyz=self._end_map_xyz,
            map_to_index=self._map_to_index,
        )
        if not planned:
            return False
        planned = self._optimize_rl_path(planned)
        return self._set_navigation_path(planned)

    def stop_navigation(self) -> None:
        self._nav_running = False
        self._nav_last_time_s = None
        if self._nav_timer.isActive():
            self._nav_timer.stop()

    def start_astar_navigation(self) -> bool:
        if self._start_map_xyz is None or self._end_map_xyz is None:
            return False

        planned = plan_astar_path_map_xyz(
            self._start_map_xyz,
            self._end_map_xyz,
            self._occupancy,
            self._occ_nx,
            self._occ_nz,
            self._heights,
            self._map_to_index,
        )
        if not planned or len(planned) < 2:
            return False
        return self._set_navigation_path(planned)

    def _sample_nav_position(self, dist: float) -> tuple[float, float, float]:
        if not self._nav_path_map_xyz:
            return (0.0, 0.0, 0.0)
        if dist <= 0.0:
            return self._nav_path_map_xyz[0]
        if dist >= self._nav_total_dist:
            return self._nav_path_map_xyz[-1]

        cum = self._nav_path_cumdist
        path = self._nav_path_map_xyz
        for i in range(1, len(cum)):
            if dist <= cum[i]:
                d0 = cum[i - 1]
                d1 = cum[i]
                if d1 <= d0 + 1e-9:
                    return path[i]
                t = (dist - d0) / (d1 - d0)
                x0, y0, z0 = path[i - 1]
                x1, y1, z1 = path[i]
                return (
                    (1.0 - t) * x0 + t * x1,
                    (1.0 - t) * y0 + t * y1,
                    (1.0 - t) * z0 + t * z1,
                )
        return path[-1]

    def _on_nav_timer(self) -> None:
        if not self._nav_running:
            return

        now = time.perf_counter()
        if self._nav_last_time_s is None:
            self._nav_last_time_s = now
            return

        dt = max(0.0, now - self._nav_last_time_s)
        self._nav_last_time_s = now

        self._nav_progress_dist += float(self._nav_speed_units_per_sec) * dt
        if self._nav_progress_dist >= self._nav_total_dist:
            self._nav_progress_dist = self._nav_total_dist
            self._agent_map_xyz = self._nav_path_map_xyz[-1]
            self.stop_navigation()
            self.update()
            return

        self._agent_map_xyz = self._sample_nav_position(self._nav_progress_dist)
        self.update()

    def add_random_buildings_map(self, count: int) -> int:
        """Add random grounded buildings and return the number actually added.

        Buildings are generated as grounded cuboids in map coordinates where
        X/Y are planar and Z is height.
        """

        if self._occupancy is None or self._occ_nx is None or self._occ_ny is None or self._occ_nz is None:
            self._emit_edit_state()
            return 0

        try:
            target_count = max(0, int(count))
        except Exception:
            return 0

        if target_count <= 0:
            return 0

        import numpy as np

        nx, ny, nz = int(self._occ_nx), int(self._occ_ny), int(self._occ_nz)
        if nx < 2 or ny < 2 or nz < 2:
            return 0

        # Keep one undo step for one random generation action.
        self._push_undo_state()

        occ = np.asarray(self._occupancy)
        rng = random.Random()

        min_span_x = 2
        min_span_y = 2
        max_span_x = max(min_span_x, nx // 6)
        max_span_y = max(min_span_y, nz // 6)
        min_h = max(2, ny // 10)
        max_h = max(min_h, ny // 2)

        placed = 0
        attempts = 0
        max_attempts = target_count * 8

        while placed < target_count and attempts < max_attempts:
            attempts += 1

            span_x = rng.randint(min_span_x, max_span_x)
            span_y = rng.randint(min_span_y, max_span_y)
            if span_x >= nx or span_y >= nz:
                continue

            ix0 = rng.randint(0, nx - span_x - 1)
            iy0 = rng.randint(0, nz - span_y - 1)
            ix1 = ix0 + span_x
            iy1 = iy0 + span_y

            height_vox = rng.randint(min_h, max_h)
            iz0 = 0
            iz1 = min(ny - 1, iz0 + height_vox)

            occ[ix0 : ix1 + 1, iz0 : iz1 + 1, iy0 : iy1 + 1] = 1
            placed += 1

        self._occupancy = occ
        self._heights = self._derive_heightfield_from_occupancy(self._occupancy, self._cell_size)
        self._invalidate_q_learning()
        self._recompile_display_list_from_current_heights()
        self._emit_edit_state()
        return placed

    def _try_build_terrain_display_list(self) -> None:
        """Load terrain from a file path and compile it into an OpenGL display list.

        Supported formats:
          - HDF5 (scheme 2 occupancy grid): dataset /occupancy (uint8) shape (nx, ny, nz)
          - CSV heightfield (legacy): 2D grid of numbers (rows x cols)
        """

        if not self._terrain_path:
            return

        p = Path(self._terrain_path)
        if not p.exists() or not p.is_file():
            return

        heights, cell_size, grid_ny, occupancy = self._load_heightfield(p)
        if heights is None:
            return

        self._heights = heights
        self._cell_size = float(cell_size)
        self._grid_rows, self._grid_cols = heights.shape
        self._grid_ny = grid_ny

        self._occupancy = occupancy
        if occupancy is not None:
            try:
                nx, ny, nz = occupancy.shape
            except Exception:
                nx = ny = nz = None
            self._occ_nx = int(nx) if nx is not None else None
            self._occ_ny = int(ny) if ny is not None else None
            self._occ_nz = int(nz) if nz is not None else None
        self._reset_edit_history()

        from OpenGL import GL as gl

        # Compile a simple triangle mesh into a display list for speed.
        self._display_list = gl.glGenLists(1)
        gl.glNewList(self._display_list, gl.GL_COMPILE)
        self._draw_terrain_mesh(gl)
        gl.glEndList()

        self._reset_camera_to_fit_terrain()

    def _load_heightfield(self, p: Path):
        """Return (heights[y,x], cell_size, grid_ny, occupancy) or (None, _, _, _).

        For scheme-2 occupancy grids, we derive a 2D surface by taking the highest
        occupied voxel along the height axis for each planar (x, y) location.
        """

        suffix = p.suffix.lower()

        # HDF5 occupancy grid
        if suffix in {".h5", ".hdf5"}:
            return load_terrain_hdf5(p)

        # CSV heightfield (legacy)
        try:
            import numpy as np

            parsed = load_occupancy_csv(p)
            if parsed is not None:
                occ, voxel_size = parsed
                nx, ny, nz = occ.shape
                if nx < 2 or ny < 2 or nz < 2:
                    return (None, voxel_size, None, None)
                heights = self._derive_heightfield_from_occupancy(occ, voxel_size)
                return (heights, voxel_size, int(ny), occ.astype(np.uint8, copy=False))

            try:
                heights = np.loadtxt(str(p), delimiter=",", dtype=float)
            except Exception:
                heights = np.loadtxt(str(p), dtype=float)

            if heights.ndim != 2:
                return (None, 1.0, None, None)
            if heights.shape[0] < 2 or heights.shape[1] < 2:
                return (None, 1.0, None, None)
            return (heights, 1.0, None, None)
        except Exception:
            return (None, 1.0, None, None)

    def _load_occupancy_csv(self, p: Path):
        """Backward-compatible wrapper for unified CSV parser."""

        return load_occupancy_csv(p)

    def _derive_heightfield_from_occupancy(self, occ, voxel_size: float):
        """Derive heights[y,x] from occupancy by taking the highest occupied height per planar cell."""

        return derive_heightfield_from_occupancy(occ, voxel_size)

    def _recompile_display_list_from_current_heights(self) -> None:
        """Recompile the terrain display list using current self._heights."""

        if not self._gl_ready or self._heights is None:
            return

        from OpenGL import GL as gl

        self.makeCurrent()
        try:
            if self._display_list is not None:
                try:
                    gl.glDeleteLists(self._display_list, 1)
                except Exception:
                    pass
                self._display_list = None

            self._display_list = gl.glGenLists(1)
            gl.glNewList(self._display_list, gl.GL_COMPILE)
            self._draw_terrain_mesh(gl)
            gl.glEndList()

            # Update stats used by camera/axes/hover mapping.
            self._grid_rows, self._grid_cols = self._heights.shape
            self._reset_camera_extents_only()
        finally:
            self.doneCurrent()

        self.update()

    def _reset_camera_extents_only(self) -> None:
        """Update extents/origin without changing view angles/distance/pan."""

        heights = self._heights
        if heights is None:
            return

        rows, cols = heights.shape
        cell = float(self._cell_size)

        half_x = (cols - 1) * cell * 0.5
        half_y = (rows - 1) * cell * 0.5

        self._origin_x = -half_x
        self._origin_y = -half_y
        self._origin_z = 0.0
        self._extent_x = max((cols - 1) * cell, 1e-9)
        self._extent_y = max((rows - 1) * cell, 1e-9)

    def _map_to_index(self, v_0_100: float, n: int, mode: str = "round") -> int:
        """Convert a 0..100 map coordinate to an index in [0, n-1]."""

        if n <= 1:
            return 0

        v = max(0.0, min(100.0, float(v_0_100)))
        t = v / 100.0 * float(n - 1)

        if mode == "floor":
            idx = int(math.floor(t))
        elif mode == "ceil":
            idx = int(math.ceil(t))
        else:
            idx = int(round(t))

        return max(0, min(n - 1, idx))

    def add_cuboid_map(self, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> bool:
        """Add an occupied cuboid specified in 0..100 map coordinates.

        Returns True on success (occupancy edited), False if no editable occupancy loaded.
        """

        if self._occupancy is None or self._occ_nx is None or self._occ_ny is None or self._occ_nz is None:
            self._emit_edit_state()
            return False

        import numpy as np

        nx, ny, nz = int(self._occ_nx), int(self._occ_ny), int(self._occ_nz)

        xa, xb = sorted([float(x0), float(x1)])
        ya, yb = sorted([float(y0), float(y1)])
        za, zb = sorted([float(z0), float(z1)])

        ix0 = self._map_to_index(xa, nx, mode="floor")
        ix1 = self._map_to_index(xb, nx, mode="ceil")
        iy0 = self._map_to_index(za, ny, mode="floor")
        iy1 = self._map_to_index(zb, ny, mode="ceil")
        iz0 = self._map_to_index(ya, nz, mode="floor")
        iz1 = self._map_to_index(yb, nz, mode="ceil")

        self._push_undo_state()
        occ = np.asarray(self._occupancy)
        occ[ix0 : ix1 + 1, iy0 : iy1 + 1, iz0 : iz1 + 1] = 1
        self._occupancy = occ
        self._heights = self._derive_heightfield_from_occupancy(self._occupancy, self._cell_size)
        self._invalidate_q_learning()
        self._recompile_display_list_from_current_heights()
        self._emit_edit_state()
        return True

    def add_cylinder_map(self, cx: float, cy: float, cz: float, radius: float, height: float) -> bool:
        """Add an occupied vertical cylinder (axis +Z) in 0..100 map coordinates."""

        if self._occupancy is None or self._occ_nx is None or self._occ_ny is None or self._occ_nz is None:
            self._emit_edit_state()
            return False

        import numpy as np

        nx, ny, nz = int(self._occ_nx), int(self._occ_ny), int(self._occ_nz)
        cx = float(cx)
        cy = float(cy)
        cz = float(cz)
        radius = max(0.0, float(radius))
        height = max(0.0, float(height))

        # Index-space center
        icx = self._map_to_index(cx, nx, mode="round")
        icy = self._map_to_index(cy, nz, mode="round")
        iy0 = self._map_to_index(cz, ny, mode="floor")
        iy1 = self._map_to_index(cz + height, ny, mode="ceil")
        if iy1 < iy0:
            iy0, iy1 = iy1, iy0

        # Radius in index units across the planar X/Y grid.
        rx = radius / 100.0 * float(nx - 1)
        ry = radius / 100.0 * float(nz - 1)
        rx = max(rx, 0.0)
        ry = max(ry, 0.0)

        self._push_undo_state()
        if rx <= 0.0 or ry <= 0.0:
            # Degenerate: treat as a single vertical line.
            self._occupancy[icx : icx + 1, iy0 : iy1 + 1, icy : icy + 1] = 1
            self._heights = self._derive_heightfield_from_occupancy(self._occupancy, self._cell_size)
            self._recompile_display_list_from_current_heights()
            self._emit_edit_state()
            return True

        ix0 = max(0, int(math.floor(icx - rx)))
        ix1 = min(nx - 1, int(math.ceil(icx + rx)))
        iz0 = max(0, int(math.floor(icy - ry)))
        iz1 = min(nz - 1, int(math.ceil(icy + ry)))

        xs = np.arange(ix0, ix1 + 1, dtype=np.float32)
        zs = np.arange(iz0, iz1 + 1, dtype=np.float32)
        xx, zz = np.meshgrid(xs, zs, indexing="xy")

        # Ellipse in index-space to approximate circle in map-space.
        dx = (xx - float(icx)) / float(rx)
        dy = (zz - float(icy)) / float(ry)
        mask = (dx * dx + dy * dy) <= 1.0

        occ = np.asarray(self._occupancy)
        # mask shape is (len(ys), len(xs)); occupancy slice expects [x, z_height, y_plane]
        for zi, z_idx in enumerate(range(iz0, iz1 + 1)):
            row_mask = mask[zi]
            if not row_mask.any():
                continue
            x_indices = np.arange(ix0, ix1 + 1)[row_mask]
            occ[x_indices, iy0 : iy1 + 1, z_idx] = 1

        self._occupancy = occ
        self._heights = self._derive_heightfield_from_occupancy(self._occupancy, self._cell_size)
        self._invalidate_q_learning()
        self._recompile_display_list_from_current_heights()
        self._emit_edit_state()
        return True

    def _reset_camera_to_fit_terrain(self) -> None:
        heights = self._heights
        if heights is None:
            return

        rows, cols = heights.shape
        cell = float(self._cell_size)

        half_x = (cols - 1) * cell * 0.5
        half_y = (rows - 1) * cell * 0.5

        # Define a map origin at the terrain corner (min x,y) in world coords.
        self._origin_x = -half_x
        self._origin_y = -half_y
        self._origin_z = 0.0
        self._extent_x = max((cols - 1) * cell, 1e-9)
        self._extent_y = max((rows - 1) * cell, 1e-9)
        if self._grid_ny is not None and self._grid_ny >= 2:
            self._extent_z = max((self._grid_ny - 1) * cell, 1e-9)
        else:
            self._extent_z = None

        hmin = float(heights.min())
        hmax = float(heights.max())
        target_z = 0.5 * (hmin + hmax)

        self._terrain_half_x = half_x
        self._terrain_half_y = half_y
        self._terrain_hmax = hmax

        self._target_x = 0.0
        self._target_y = 0.0
        self._target_z = target_z

        # Distance based on terrain size; ensure it's not too close.
        radius = max(half_x, half_y, 1.0)
        self._distance = radius * 2.2

        # Default view angles similar to MATLAB.
        self._yaw = 45.0
        self._pitch = -35.0

        self._pan_x = 0.0
        self._pan_y = 0.0

    def _draw_terrain_mesh(self, gl) -> None:
        heights = self._heights
        if heights is None:
            return

        rows, cols = heights.shape
        cell = float(self._cell_size)

        # Center the terrain around origin.
        half_x = (cols - 1) * cell * 0.5
        half_y = (rows - 1) * cell * 0.5

        hmin = float(heights.min())
        hmax = float(heights.max())
        span = hmax - hmin

        # Interpret CSV values as absolute heights (world units).
        # You can change this factor later if your CSV units differ.
        height_scale = 1.0

        def color_for_h(h: float) -> tuple[float, float, float]:
            if span <= 1e-9:
                t = 0.5
            else:
                t = (h - hmin) / span
            # Simple green-to-yellow-to-white ramp.
            r = 0.15 + 0.75 * t
            g = 0.35 + 0.55 * t
            b = 0.15 + 0.20 * (1.0 - t)
            return (r, g, b)

        gl.glBegin(gl.GL_TRIANGLES)
        for j in range(rows - 1):
            for i in range(cols - 1):
                # Grid coords -> world
                x0 = i * cell - half_x
                x1 = (i + 1) * cell - half_x
                y0 = j * cell - half_y
                y1 = (j + 1) * cell - half_y

                h00 = float(heights[j, i])
                h10 = float(heights[j, i + 1])
                h01 = float(heights[j + 1, i])
                h11 = float(heights[j + 1, i + 1])

                z00 = h00 * height_scale
                z10 = h10 * height_scale
                z01 = h01 * height_scale
                z11 = h11 * height_scale

                # Triangle 1: (00, 10, 11)
                gl.glColor3f(*color_for_h(h00))
                gl.glVertex3f(x0, y0, z00)
                gl.glColor3f(*color_for_h(h10))
                gl.glVertex3f(x1, y0, z10)
                gl.glColor3f(*color_for_h(h11))
                gl.glVertex3f(x1, y1, z11)

                # Triangle 2: (00, 11, 01)
                gl.glColor3f(*color_for_h(h00))
                gl.glVertex3f(x0, y0, z00)
                gl.glColor3f(*color_for_h(h11))
                gl.glVertex3f(x1, y1, z11)
                gl.glColor3f(*color_for_h(h01))
                gl.glVertex3f(x0, y1, z01)
        gl.glEnd()

    def resizeGL(self, w: int, h: int) -> None:  # noqa: N802
        from OpenGL import GL as gl
        from OpenGL.GLU import gluPerspective

        h = max(h, 1)
        gl.glViewport(0, 0, w, h)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # Wide enough z-range for terrain.
        gluPerspective(45.0, float(w) / float(h), 0.1, 10000.0)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def _pick_map_xy_at_screen_pos(self, pos) -> tuple[float, float] | None:
        """Pick map (mx,my) under a screen position using depth-buffer unprojection."""

        if not self._gl_ready or self._display_list is None:
            return None

        try:
            from OpenGL import GL as gl
            from OpenGL import GLU as glu
        except Exception:
            return None

        ratio = float(self.devicePixelRatioF())
        x = int(pos.x() * ratio)
        y = int((self.height() - 1 - pos.y()) * ratio)
        if x < 0 or y < 0:
            return None

        self.makeCurrent()
        try:
            viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
            vw = int(viewport[2])
            vh = int(viewport[3])
            if x >= vw or y >= vh:
                return None

            depth = gl.glReadPixels(x, y, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)

            if isinstance(depth, (bytes, bytearray)):
                import struct

                z = struct.unpack("f", depth[:4])[0]
            elif isinstance(depth, (float, int)):
                z = float(depth)
            else:
                try:
                    z = float(depth[0][0])
                except Exception:
                    try:
                        z = float(depth[0])
                    except Exception:
                        return None

            if z >= 0.999999:
                return None

            model = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
            proj = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
            wx, wy, _wz = glu.gluUnProject(x, y, z, model, proj, viewport)

            mx = (float(wx) - self._origin_x) / max(self._extent_x, 1e-9) * 100.0
            my = (float(wy) - self._origin_y) / max(self._extent_y, 1e-9) * 100.0
            mx = max(0.0, min(100.0, mx))
            my = max(0.0, min(100.0, my))
            return (mx, my)
        finally:
            self.doneCurrent()

    def _project_world_to_widget(self, wx: float, wy: float, wz: float) -> tuple[float, float] | None:
        """Project a world point to widget pixel coords using cached GL matrices."""

        if self._last_gl_model is None or self._last_gl_proj is None or self._last_gl_viewport is None:
            return None

        m = self._last_gl_model
        p = self._last_gl_proj
        vx, vy, vw, vh = self._last_gl_viewport

        def mul4(mat: tuple[float, ...], v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
            x, y, z, w = v
            r0 = mat[0] * x + mat[4] * y + mat[8] * z + mat[12] * w
            r1 = mat[1] * x + mat[5] * y + mat[9] * z + mat[13] * w
            r2 = mat[2] * x + mat[6] * y + mat[10] * z + mat[14] * w
            r3 = mat[3] * x + mat[7] * y + mat[11] * z + mat[15] * w
            return (r0, r1, r2, r3)

        v = (float(wx), float(wy), float(wz), 1.0)
        eye = mul4(m, v)
        clip = mul4(p, eye)
        if abs(clip[3]) <= 1e-12:
            return None

        ndc_x = clip[0] / clip[3]
        ndc_y = clip[1] / clip[3]

        win_x = float(vx) + (ndc_x + 1.0) * 0.5 * float(vw)
        win_y = float(vy) + (ndc_y + 1.0) * 0.5 * float(vh)

        ratio = float(self._last_gl_ratio) if self._last_gl_ratio > 0 else 1.0
        px = win_x / ratio
        py = (float(vh) - win_y) / ratio
        return (px, py)

    @staticmethod
    def _coerce_gl_matrix_16(mat) -> tuple[float, ...] | None:
        """Convert various PyOpenGL matrix return types to a 16-float tuple."""

        if mat is None:
            return None

        # Common cases: numpy arrays (have flatten), array.array, lists, nested lists.
        try:
            flat = mat.flatten()  # type: ignore[attr-defined]
            values = [float(x) for x in flat]
        except Exception:
            try:
                values = [float(x) for x in mat]
            except Exception:
                try:
                    values = []
                    for row in mat:
                        for x in row:
                            values.append(float(x))
                except Exception:
                    return None

        if len(values) != 16:
            return None
        return tuple(values)

    def paintGL(self) -> None:  # noqa: N802
        from OpenGL import GL as gl
        from OpenGL import GLU as glu
        from OpenGL.GLU import gluPerspective

        # QPainter overlays (paintEvent) can mutate GL state (notably the projection matrix).
        # Re-apply viewport + projection every frame to keep perspective and picking stable.
        ratio = float(self.devicePixelRatioF())
        w_px = max(1, int(self.width() * ratio))
        h_px = max(1, int(self.height() * ratio))
        gl.glViewport(0, 0, w_px, h_px)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gluPerspective(45.0, float(w_px) / float(h_px), 0.1, 10000.0)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        # Orbit camera around target (MATLAB-like).
        yaw = math.radians(self._yaw)
        pitch = math.radians(self._pitch)

        cx = math.cos(pitch)
        sx = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        # Forward direction from camera to target
        fx = cx * cy
        fy = cx * sy
        fz = sx

        # Camera position
        tx = self._target_x + self._pan_x
        ty = self._target_y + self._pan_y
        tz = self._target_z

        ex = tx - fx * self._distance
        ey = ty - fy * self._distance
        ez = tz - fz * self._distance

        glu.gluLookAt(ex, ey, ez, tx, ty, tz, 0.0, 0.0, 1.0)

        # Cache matrices for projection-based deletion and overlay.
        try:
            viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
            model = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
            proj = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
            self._last_gl_viewport = (int(viewport[0]), int(viewport[1]), int(viewport[2]), int(viewport[3]))
            self._last_gl_model = self._coerce_gl_matrix_16(model)
            self._last_gl_proj = self._coerce_gl_matrix_16(proj)
            self._last_gl_ratio = float(self.devicePixelRatioF())
        except Exception:
            self._last_gl_viewport = None
            self._last_gl_model = None
            self._last_gl_proj = None

        # Cache per-pin screen positions using GLU (robust across matrix formats).
        self._pin_screen_xy = []
        try:
            vp = gl.glGetIntegerv(gl.GL_VIEWPORT)
            vp_h = int(vp[3])
            ratio = float(self.devicePixelRatioF())
            mdl = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
            prj = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)

            for mx, my in self._pinned_xy:
                wx = self._origin_x + (float(mx) / 100.0) * float(self._extent_x)
                wy = self._origin_y + (float(my) / 100.0) * float(self._extent_y)
                wz = self._terrain_height_at_map(mx, my)
                try:
                    sx, sy, _sz = glu.gluProject(float(wx), float(wy), float(wz), mdl, prj, vp)
                    px = float(sx) / ratio
                    py = float(vp_h - sy) / ratio
                    if not (math.isfinite(px) and math.isfinite(py)):
                        self._pin_screen_xy.append(None)
                    else:
                        self._pin_screen_xy.append((px, py))
                except Exception:
                    self._pin_screen_xy.append(None)
        except Exception:
            self._pin_screen_xy = [None for _ in self._pinned_xy]

        # Draw terrain if available.
        if self._display_list is not None:
            gl.glCallList(self._display_list)
            self._draw_origin_axes(gl)
            self._draw_pinned_points(gl)
            self._draw_trace_points(gl)
            self._draw_navigation_path(gl)
            self._draw_agent_ball(gl, glu)
            self._update_hover_xyz(gl, glu)
        else:
            # Draw a simple colored cube using immediate mode (fallback demo).
            gl.glBegin(gl.GL_QUADS)

            # Front (z+)
            gl.glColor3f(0.9, 0.2, 0.2)
            gl.glVertex3f(-1, -1, 1)
            gl.glVertex3f(1, -1, 1)
            gl.glVertex3f(1, 1, 1)
            gl.glVertex3f(-1, 1, 1)

            # Back (z-)
            gl.glColor3f(0.2, 0.9, 0.2)
            gl.glVertex3f(-1, -1, -1)
            gl.glVertex3f(-1, 1, -1)
            gl.glVertex3f(1, 1, -1)
            gl.glVertex3f(1, -1, -1)

            # Left (x-)
            gl.glColor3f(0.2, 0.2, 0.9)
            gl.glVertex3f(-1, -1, -1)
            gl.glVertex3f(-1, -1, 1)
            gl.glVertex3f(-1, 1, 1)
            gl.glVertex3f(-1, 1, -1)

            # Right (x+)
            gl.glColor3f(0.9, 0.9, 0.2)
            gl.glVertex3f(1, -1, -1)
            gl.glVertex3f(1, 1, -1)
            gl.glVertex3f(1, 1, 1)
            gl.glVertex3f(1, -1, 1)

            # Top (y+)
            gl.glColor3f(0.2, 0.9, 0.9)
            gl.glVertex3f(-1, 1, -1)
            gl.glVertex3f(-1, 1, 1)
            gl.glVertex3f(1, 1, 1)
            gl.glVertex3f(1, 1, -1)

            # Bottom (y-)
            gl.glColor3f(0.9, 0.2, 0.9)
            gl.glVertex3f(-1, -1, -1)
            gl.glVertex3f(1, -1, -1)
            gl.glVertex3f(1, -1, 1)
            gl.glVertex3f(-1, -1, 1)

            gl.glEnd()

            self._draw_origin_axes(gl)
            self._draw_pinned_points(gl)
            self._draw_trace_points(gl)
            self._draw_navigation_path(gl)
            self._draw_agent_ball(gl, glu)
            self._update_hover_xyz(gl, glu)

        self._update_pin_labels()
        self._update_trace_labels(gl, glu)

    def _draw_origin_axes(self, gl) -> None:
        """Draw XYZ axes at the map origin (terrain corner) in world coordinates."""

        if not self._show_axes:
            return

        # Scale axes length to be visible across typical 100x100 maps.
        axis_len = 0.35 * max(self._extent_x, self._extent_y, 1.0)

        ox = float(self._origin_x)
        oy = float(self._origin_y)
        oz = float(self._origin_z)

        gl.glLineWidth(3.0)
        gl.glBegin(gl.GL_LINES)

        # X axis (red)
        gl.glColor3f(0.9, 0.2, 0.2)
        gl.glVertex3f(ox, oy, oz)
        gl.glVertex3f(ox + axis_len, oy, oz)

        # Y axis (green)
        gl.glColor3f(0.2, 0.9, 0.2)
        gl.glVertex3f(ox, oy, oz)
        gl.glVertex3f(ox, oy + axis_len, oz)

        # Z axis (blue, height)
        gl.glColor3f(0.2, 0.4, 0.95)
        gl.glVertex3f(ox, oy, oz)
        gl.glVertex3f(ox, oy, oz + axis_len)

        gl.glEnd()

    def _draw_pinned_points(self, gl) -> None:
        if not self._pinned_xy or self._heights is None:
            return

        # Draw pins slightly above the terrain to avoid z-fighting.
        z_eps = 0.002 * max(self._extent_x, self._extent_y, 1.0)

        gl.glDisable(gl.GL_LIGHTING)
        gl.glPointSize(8.0)
        gl.glBegin(gl.GL_POINTS)
        for mx, my in self._pinned_xy:
            wx = self._origin_x + (float(mx) / 100.0) * float(self._extent_x)
            wy = self._origin_y + (float(my) / 100.0) * float(self._extent_y)
            wz = self._terrain_height_at_map(mx, my) + z_eps
            gl.glColor3f(1.0, 0.25, 1.0)
            gl.glVertex3f(float(wx), float(wy), float(wz))
        gl.glEnd()

    def _draw_trace_points(self, gl) -> None:
        if self._start_map_xyz is None and self._end_map_xyz is None:
            return

        z_eps = 0.002 * max(self._extent_x, self._extent_y, 1.0)

        gl.glDisable(gl.GL_LIGHTING)
        gl.glPointSize(11.0)
        gl.glBegin(gl.GL_POINTS)

        if self._start_map_xyz is not None:
            sx, sy, sz = self._start_map_xyz
            wx, wy, wz = self._map_xyz_to_world(sx, sy, sz)
            gl.glColor3f(0.2, 1.0, 0.75)
            gl.glVertex3f(wx, wy, wz + z_eps)

        if self._end_map_xyz is not None:
            ex, ey, ez = self._end_map_xyz
            wx, wy, wz = self._map_xyz_to_world(ex, ey, ez)
            gl.glColor3f(1.0, 0.6, 0.2)
            gl.glVertex3f(wx, wy, wz + z_eps)

        gl.glEnd()

    def _draw_agent_ball(self, gl, glu) -> None:
        if self._agent_map_xyz is None:
            return

        ax, ay, az = self._agent_map_xyz
        wx, wy, wz = self._map_xyz_to_world(ax, ay, az)

        radius = max(0.35, 0.01 * max(self._extent_x, self._extent_y, 1.0))

        was_lighting = bool(gl.glIsEnabled(gl.GL_LIGHTING))
        was_light0 = bool(gl.glIsEnabled(gl.GL_LIGHT0))

        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)

        # Local light near the agent to make the sphere shape pop.
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (wx + 3.0 * radius, wy + 2.5 * radius, wz + 5.0 * radius, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0.20, 0.20, 0.20, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (0.95, 0.95, 0.95, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, (1.00, 1.00, 1.00, 1.0))

        # Red glossy material for the agent ball.
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE, (0.88, 0.14, 0.14, 1.0))
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, (1.00, 0.85, 0.85, 1.0))
        gl.glMaterialf(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, 56.0)

        gl.glPushMatrix()
        gl.glTranslatef(float(wx), float(wy), float(wz))
        quad = glu.gluNewQuadric()
        try:
            glu.gluSphere(quad, float(radius), 20, 20)
        finally:
            try:
                glu.gluDeleteQuadric(quad)
            except Exception:
                pass
            gl.glPopMatrix()

        if not was_light0:
            gl.glDisable(gl.GL_LIGHT0)
        if not was_lighting:
            gl.glDisable(gl.GL_LIGHTING)

    def _draw_navigation_path(self, gl) -> None:
        if len(self._nav_path_map_xyz) < 2:
            return

        gl.glDisable(gl.GL_LIGHTING)
        gl.glLineWidth(2.5)
        try:
            gl.glEnable(gl.GL_LINE_STIPPLE)
            gl.glLineStipple(1, 0x00FF)
        except Exception:
            pass

        gl.glColor3f(0.25, 0.55, 1.0)
        gl.glBegin(gl.GL_LINE_STRIP)
        for mx, my, mz in self._nav_path_map_xyz:
            wx, wy, wz = self._map_xyz_to_world(mx, my, mz)
            gl.glVertex3f(float(wx), float(wy), float(wz))
        gl.glEnd()

        try:
            gl.glDisable(gl.GL_LINE_STIPPLE)
        except Exception:
            pass

    def _update_hover_xyz(self, gl, glu) -> None:
        """If hovering, read depth buffer under cursor and unproject to world XYZ."""

        if self._hover_pos is None:
            return

        # Convert Qt coords (top-left origin) to GL window coords (bottom-left).
        ratio = float(self.devicePixelRatioF())
        x = int(self._hover_pos.x() * ratio)
        y = int((self.height() - 1 - self._hover_pos.y()) * ratio)

        if x < 0 or y < 0:
            return

        try:
            viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
            vw = int(viewport[2])
            vh = int(viewport[3])
            if x >= vw or y >= vh:
                return

            depth = gl.glReadPixels(x, y, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)

            # PyOpenGL may return bytes, a float, or a nested sequence.
            if isinstance(depth, (bytes, bytearray)):
                import struct

                z = struct.unpack("f", depth[:4])[0]
            elif isinstance(depth, (float, int)):
                z = float(depth)
            else:
                try:
                    z = float(depth[0][0])
                except Exception:
                    try:
                        z = float(depth[0])
                    except Exception:
                        return

            # z==1.0 means background (no geometry drawn there).
            if z >= 0.999999:
                self._last_hover_map_xy = None
                self._emit_xyz_text("XYZ: --")
                return

            model = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
            proj = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)

            wx, wy, wz = glu.gluUnProject(x, y, z, model, proj, viewport)

            # Convert from centered world coords to map coords with origin at terrain corner.
            # Map coordinate range is normalized to 0..100 (inclusive endpoints).
            mx = (float(wx) - self._origin_x) / max(self._extent_x, 1e-9) * 100.0
            my = (float(wy) - self._origin_y) / max(self._extent_y, 1e-9) * 100.0
            mx = max(0.0, min(100.0, mx))
            my = max(0.0, min(100.0, my))

            self._last_hover_map_xy = (mx, my)

            if self._extent_z is not None and self._extent_z > 0:
                mz = float(wz) / self._extent_z * 100.0
                mz = max(0.0, min(100.0, mz))
                self._emit_xyz_text(f"XYZ: {mx:.1f}, {my:.1f}, {mz:.1f}")
            else:
                self._emit_xyz_text(f"XYZ: {mx:.1f}, {my:.1f}, {wz:.2f}")
        except Exception:
            # Don't spam; just show unknown.
            self._last_hover_map_xy = None
            self._emit_xyz_text("XYZ: --")

    def _ensure_pin_labels_count(self, n: int) -> None:
        while len(self._pin_labels) < n:
            lb = QLabel(self)
            lb.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            lb.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 160); padding: 2px 4px;")
            lb.hide()
            self._pin_labels.append(lb)

        while len(self._pin_labels) > n:
            lb = self._pin_labels.pop()
            lb.hide()
            lb.deleteLater()

    def _update_pin_labels(self) -> None:
        n = len(self._pinned_xy)
        self._ensure_pin_labels_count(n)

        for i in range(n):
            lb = self._pin_labels[i]
            proj = self._pin_screen_xy[i] if i < len(self._pin_screen_xy) else None
            if proj is None:
                lb.hide()
                continue

            px, py = proj
            if not (math.isfinite(px) and math.isfinite(py)):
                lb.hide()
                continue

            mx, my = self._pinned_xy[i]
            text = self._format_xyz_text_from_map(mx, my).replace("XYZ: ", "")
            lb.setText(text)
            lb.adjustSize()

            x0 = int(px + 8)
            y0 = int(py - lb.height() - 8)

            # Keep label inside widget bounds.
            x0 = max(0, min(max(0, self.width() - lb.width()), x0))
            y0 = max(0, min(max(0, self.height() - lb.height()), y0))

            lb.move(x0, y0)
            lb.show()
            lb.raise_()

    def _update_trace_labels(self, gl, glu) -> None:
        ratio = float(self.devicePixelRatioF())

        try:
            vp = gl.glGetIntegerv(gl.GL_VIEWPORT)
            vp_h = int(vp[3])
            mdl = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
            prj = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
        except Exception:
            self._start_label.hide()
            self._end_label.hide()
            return

        def place_label(lb: QLabel, prefix: str, xyz_map: tuple[float, float, float] | None) -> None:
            if xyz_map is None:
                lb.hide()
                return

            mx, my, mz = xyz_map
            wx, wy, wz = self._map_xyz_to_world(mx, my, mz)
            try:
                sx, sy, _sz = glu.gluProject(float(wx), float(wy), float(wz), mdl, prj, vp)
            except Exception:
                lb.hide()
                return

            px = float(sx) / ratio
            py = float(vp_h - sy) / ratio
            if not (math.isfinite(px) and math.isfinite(py)):
                lb.hide()
                return

            lb.setText(f"{prefix} {mx:.1f},{my:.1f},{mz:.1f}")
            lb.adjustSize()

            x0 = int(px + 8)
            y0 = int(py - lb.height() - 8)
            x0 = max(0, min(max(0, self.width() - lb.width()), x0))
            y0 = max(0, min(max(0, self.height() - lb.height()), y0))
            lb.move(x0, y0)
            lb.show()
            lb.raise_()

        place_label(self._start_label, "S:", self._start_map_xyz)
        place_label(self._end_label, "E:", self._end_map_xyz)


    def mousePressEvent(self, event):  # type: ignore[override]
        self.setFocus()
        self._press_pos = event.position()
        self._press_button = event.button()
        self._press_dragged = False

        self._last_mouse_pos = event.position()
        self._hover_pos = event.position()
        if event.button() == Qt.LeftButton:
            self._pending_mouse_mode = "orbit"
        elif event.button() == Qt.MiddleButton:
            self._pending_mouse_mode = "pan"
        else:
            self._pending_mouse_mode = None
        self._mouse_mode = None

        self.update()

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        # Click actions (pin / delete) are recognized on release if the mouse didn't drag.
        if self._press_pos is not None and not self._press_dragged:
            if self._press_button == Qt.LeftButton:
                picked = self._last_hover_map_xy
                if picked is None:
                    picked = self._pick_map_xy_at_screen_pos(event.position())
                if picked is not None:
                    self._pinned_xy.append(picked)
                    self._active_pin_index = len(self._pinned_xy) - 1
                    mx, my = picked
                    self._emit_xyz_text(self._format_xyz_text_from_map(mx, my))
            elif self._press_button == Qt.RightButton:
                if self._pinned_xy:
                    click = event.position()
                    cx = float(click.x())
                    cy = float(click.y())

                    best_i = None
                    best_d = None

                    for i in range(len(self._pinned_xy)):
                        if i >= len(self._pin_screen_xy):
                            continue
                        proj = self._pin_screen_xy[i]
                        if proj is None:
                            continue
                        px, py = proj
                        d = math.hypot(px - cx, py - cy)
                        if best_d is None or d < best_d:
                            best_d = d
                            best_i = i

                    # Threshold in pixels.
                    if best_i is not None and best_d is not None and best_d <= 14.0:
                        self._pinned_xy.pop(int(best_i))
                        if not self._pinned_xy:
                            self._active_pin_index = None
                            self._emit_xyz_text("XYZ: --")
                        else:
                            self._active_pin_index = len(self._pinned_xy) - 1
                            mx2, my2 = self._pinned_xy[self._active_pin_index]
                            self._emit_xyz_text(self._format_xyz_text_from_map(mx2, my2))

        self._mouse_mode = None
        self._pending_mouse_mode = None
        self._last_mouse_pos = None
        self._press_pos = None
        self._press_button = None
        self._press_dragged = False

        self.update()

    def mouseMoveEvent(self, event):  # type: ignore[override]
        self._hover_pos = event.position()

        if self._press_pos is not None and not self._press_dragged:
            dxp = float(event.position().x() - self._press_pos.x())
            dyp = float(event.position().y() - self._press_pos.y())
            if math.hypot(dxp, dyp) >= float(self._click_drag_threshold_px):
                self._press_dragged = True

        # Update hover readout even if not dragging.
        if self._last_mouse_pos is None:
            self.update()
            return

        # If we have a pending mode, only engage it after a real drag.
        if self._mouse_mode is None and self._pending_mouse_mode is not None:
            if not self._press_dragged:
                self.update()
                return
            self._mouse_mode = self._pending_mouse_mode
            self._pending_mouse_mode = None
            self._last_mouse_pos = event.position()
            self.update()
            return

        if self._mouse_mode is None:
            self.update()
            return

        pos = event.position()
        dx = float(pos.x() - self._last_mouse_pos.x())
        dy = float(pos.y() - self._last_mouse_pos.y())
        self._last_mouse_pos = pos

        if self._mouse_mode == "orbit":
            self._yaw = (self._yaw + dx * 0.35) % 360.0
            self._pitch = max(-89.0, min(89.0, self._pitch + dy * 0.35))
            self.update()
            return

        if self._mouse_mode == "pan":
            # Pan in screen-space roughly proportional to distance.
            pan_scale = max(self._distance, 1.0) * 0.002
            self._pan_x -= dx * pan_scale
            self._pan_y += dy * pan_scale
            self.update()

    def wheelEvent(self, event):  # type: ignore[override]
        delta = event.angleDelta().y()
        if delta == 0:
            return
        zoom = 0.9 if delta > 0 else 1.1
        self._distance = max(0.5, self._distance * zoom)
        self.update()

    def keyPressEvent(self, event):  # type: ignore[override]
        key = event.key()
        step = float(self._agent_step_map)

        moved = False
        if key == Qt.Key_W or key == Qt.Key_Up:
            moved = self.move_agent_map(0.0, step, 0.0)
        elif key == Qt.Key_S or key == Qt.Key_Down:
            moved = self.move_agent_map(0.0, -step, 0.0)
        elif key == Qt.Key_A or key == Qt.Key_Left:
            moved = self.move_agent_map(-step, 0.0, 0.0)
        elif key == Qt.Key_D or key == Qt.Key_Right:
            moved = self.move_agent_map(step, 0.0, 0.0)
        elif key == Qt.Key_R or key == Qt.Key_PageUp:
            moved = self.move_agent_map(0.0, 0.0, step)
        elif key == Qt.Key_F or key == Qt.Key_PageDown:
            moved = self.move_agent_map(0.0, 0.0, -step)

        if moved:
            event.accept()
            return

        if key == Qt.Key_Space:
            if self._nav_running:
                self.stop_navigation()
            else:
                self.start_astar_navigation()
            event.accept()
            return

        super().keyPressEvent(event)
