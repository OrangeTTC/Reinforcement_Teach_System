"""Qt integration without displaying windows or changing user settings."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from algorithm.flight_constraints import FlightConstraints
from map.map_qt_3d_view import ThreeDWindow
from map.map_qt_opengl_widget import RotatingCubeWidget


class FlightUIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.widget = RotatingCubeWidget()
        self.widget._occupancy = np.zeros((41, 41, 41), dtype=np.uint8)
        self.widget._occupancy[:, 0, :] = 1
        self.widget._occupancy[18:23, :27, 17:24] = 1
        self.widget._occ_nx = self.widget._occ_ny = self.widget._occ_nz = 41
        self.config = FlightConstraints(cruise_speed=3, lateral_accel=3, climb_rate=1.5)
        self.widget.set_flight_constraints(self.config)
        self.widget.set_trace_points_map(10, 50, 35, 90, 50, 35)

    def tearDown(self) -> None:
        self.widget.stop_navigation()
        self.widget.close()
        self.widget.deleteLater()
        self.app.processEvents()

    def test_astar_distance_acceleration_replay_and_reset(self) -> None:
        with patch("map.map_qt_opengl_widget.time.perf_counter", return_value=0):
            self.assertTrue(self.widget.start_astar_navigation())
        length = self.widget.get_last_navigation_length()
        raw = self.widget.get_last_planning_polyline()
        self.assertLess(len(raw), len(self.widget.get_last_navigation_path()))
        with patch("map.map_qt_opengl_widget.time.perf_counter", return_value=.5):
            self.widget._on_nav_timer()
        self.assertAlmostEqual(self.widget._nav_progress_dist, .25)
        np.testing.assert_allclose(self.widget._agent_map_xyz, self.widget._flight_trajectory.position_map(.25))
        self.assertTrue(self.widget._set_navigation_path(raw))
        self.assertAlmostEqual(self.widget.get_last_navigation_length(), length, places=8)
        self.assertTrue(self.widget.reset_agent_to_start())
        self.assertEqual(self.widget.get_last_planning_polyline(), [])
        self.assertIsNone(self.widget._flight_trajectory)

    def test_switch_off_restores_point_planning_and_invalidates_models(self) -> None:
        ok, msg = self.widget.train_navigation_policy("Q-learning")
        self.assertTrue(ok, msg)
        self.assertTrue(self.widget.is_algorithm_trained("Q-learning"))
        self.widget.set_flight_constraints(None)
        self.assertFalse(self.widget.is_algorithm_trained("Q-learning"))
        self.assertTrue(self.widget.start_astar_navigation())
        self.assertIsNone(self.widget._flight_trajectory)
        self.assertEqual(self.widget.get_navigation_comparison_key()[0], "u")

    def test_parameter_change_during_training_cannot_restore_stale_model(self) -> None:
        def mutate(progress: int) -> None:
            if self.widget._flight_constraints is not None:
                self.widget.set_flight_constraints(None)
        ok, msg = self.widget.train_navigation_policy("Q-learning", mutate)
        self.assertFalse(ok)
        self.assertIn("重新训练", msg)
        self.assertFalse(self.widget.is_algorithm_trained("Q-learning"))

    def test_ui_switch_enables_controls_and_uses_cruise_speed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings = QSettings(str(Path(folder) / "settings.ini"), QSettings.IniFormat)
            with patch("map.map_qt_3d_view.QSettings", return_value=settings):
                window = ThreeDWindow()
            try:
                self.assertFalse(window.constraints_checkbox.isChecked())
                self.assertFalse(window.cruise_speed.isEnabled())
                window.constraints_checkbox.setChecked(True)
                self.assertTrue(window.cruise_speed.isEnabled())
                self.assertFalse(window.navigate_speed.isEnabled())
                window.cruise_speed.setValue(4)
                self.assertEqual(window._gl._flight_constraints.cruise_speed, 4)
                window.max_speed_limit.setValue(2)
                self.assertIn("不能超过", window.feasibility_status.text())
                window.constraints_checkbox.setChecked(False)
                self.assertIsNone(window._gl._flight_constraints)
            finally:
                window._gl.stop_navigation()
                window.close()
                window.deleteLater()
                self.app.processEvents()

    def test_results_record_motion_time_and_replay_exact_speed_profile(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            settings=QSettings(str(Path(folder)/'settings.ini'),QSettings.IniFormat)
            with patch('map.map_qt_3d_view.QSettings',return_value=settings):
                window=ThreeDWindow()
            try:
                gl=window._gl
                gl._occupancy=self.widget._occupancy.copy()
                gl._occ_nx=gl._occ_ny=gl._occ_nz=41
                window.constraints_checkbox.setChecked(True)
                for edit,value in zip([window.sx_edit,window.sy_edit,window.sz_edit,
                                       window.ex_edit,window.ey_edit,window.ez_edit],
                                      [10,50,35,90,50,35]):
                    edit.setText(str(value))
                window.set_trace_button.click()
                window.algo_combo.setCurrentText('Q-learning')
                window.train_algo_button.click()
                window.navigate_button.click()
                self.assertEqual(window.metrics_table.columnCount(),7)
                self.assertEqual(len(window._metrics_history),1)
                record=window._metrics_history[0]
                trajectory=record['trajectory']
                self.assertAlmostEqual(record['flight_time'],trajectory.duration)
                self.assertEqual(window.metrics_table.item(0,6).text(),f'{trajectory.duration:.3f}')
                window.metrics_table.cellClicked.emit(0,0)
                self.assertIs(gl.get_last_flight_trajectory(),trajectory)
                np.testing.assert_allclose(gl._flight_trajectory.speed_profile.speeds,trajectory.speed_profile.speeds)
                window.cruise_speed.setValue(4)
                window.metrics_table.cellClicked.emit(0,0)
                self.assertIn('已改变',window.trace_status.text())
                self.assertFalse(gl.replay_navigation_path(record['planning_path'],trajectory))
                self.assertIn('不匹配',gl.get_last_feasibility_message())
            finally:
                window._gl.stop_navigation()
                window.close()
                window.deleteLater()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
