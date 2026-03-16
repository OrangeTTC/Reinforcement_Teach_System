from __future__ import annotations

from pathlib import Path
from time import perf_counter

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ThreeDWindow(QMainWindow):
    """A minimal in-window 3D animation using QOpenGLWidget + PyOpenGL."""

    def __init__(self, terrain_path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("3D View (OpenGL)")
        self.setMinimumSize(1000, 700)

        self._settings = QSettings("RL3D", "NavigationDemo")

        from qt_opengl_widget import RotatingCubeWidget

        resolved_path: str | None = None
        if terrain_path:
            p = Path(terrain_path)
            # Interpret relative path as "relative to workspace" (this file's directory).
            if not p.is_absolute():
                p = Path(__file__).resolve().parent / p
            resolved_path = str(p)

        root = QWidget()
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left_panel.setMinimumWidth(360)
        left_panel.setMaximumWidth(460)

        top_bar = QWidget()
        top_layout = QVBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        algo_label = QLabel("Algorithm")
        algo_label.setMinimumWidth(70)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["A*", "Q-learning", "DQN", "PPO", "HER"])
        self.algo_combo.setCurrentIndex(0)
        self.algo_combo.setMinimumWidth(170)

        self.train_algo_button = QPushButton("Train")
        self.train_algo_button.setMinimumWidth(90)
        self.algo_status = QLabel("")
        self.algo_progress = QProgressBar()
        self.algo_progress.setRange(0, 100)
        self.algo_progress.setValue(0)
        self.algo_progress.setTextVisible(True)
        self.algo_progress.setFormat("%p%")

        terrain_label = QLabel("Terrain")
        terrain_label.setMinimumWidth(70)

        self.terrain_combo = QComboBox()
        self.terrain_combo.setMinimumWidth(170)

        # Scheme 2: HDF5 occupancy-grid maps.
        self._terrain_options: dict[str, str] = {
            "Flat (height=10)": "map_flat.h5",
            "Mountain": "map_mountain.h5",
        }
        self.terrain_combo.addItems(list(self._terrain_options.keys()))

        saved_terrain = str(self._settings.value("terrain/key", "Flat (height=10)"))
        if saved_terrain in self._terrain_options:
            self.terrain_combo.setCurrentText(saved_terrain)
        else:
            self.terrain_combo.setCurrentIndex(0)

        self.clear_terrain_setting_button = QPushButton("Clear Terrain Setting")
        self.clear_terrain_setting_button.setMinimumWidth(140)
        self.export_terrain_button = QPushButton("Export Terrain CSV")
        self.export_terrain_button.setMinimumWidth(120)
        self.import_terrain_button = QPushButton("Import Terrain CSV")
        self.import_terrain_button.setMinimumWidth(120)

        default_terrain_key = "Flat (height=10)"

        def edited_terrain_path(key: str) -> Path | None:
            rel = self._terrain_options.get(key)
            if not rel:
                return None
            base_name = Path(rel).stem
            return Path(__file__).resolve().parent / "saved_terrains" / f"{base_name}_edited.h5"

        def resolved_terrain_path_for_key(key: str) -> str | None:
            rel = self._terrain_options.get(key)
            if not rel:
                return None

            edited = edited_terrain_path(key)
            if edited is not None and edited.exists():
                return str(edited)

            return resolve_path(rel)

        def resolve_path(rel_or_abs: str) -> str:
            p2 = Path(rel_or_abs)
            if not p2.is_absolute():
                p2 = Path(__file__).resolve().parent / p2
            return str(p2)

        # If caller did not override terrain_path, load saved terrain as startup map.
        if resolved_path is None:
            resolved_path = resolved_terrain_path_for_key(self.terrain_combo.currentText())

        def on_terrain_changed() -> None:
            key = self.terrain_combo.currentText()
            path = resolved_terrain_path_for_key(key)
            if not path:
                return
            self._gl.set_terrain_path(path)
            self._settings.setValue("terrain/key", key)
            update_navigation_ui()

        def persist_active_terrain() -> bool:
            key = self.terrain_combo.currentText()
            save_path = edited_terrain_path(key)
            if save_path is None:
                return False
            return self._gl.save_current_terrain_hdf5(str(save_path))

        def on_clear_terrain_setting() -> None:
            # Reset startup terrain preference.
            self._settings.remove("terrain/key")

            # Remove all persisted edited terrains and fall back to built-in maps.
            for key in self._terrain_options:
                p = edited_terrain_path(key)
                if p is not None and p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass

            self.terrain_combo.blockSignals(True)
            self.terrain_combo.setCurrentText(default_terrain_key)
            self.terrain_combo.blockSignals(False)

            path = resolved_terrain_path_for_key(default_terrain_key)
            if path:
                self._gl.set_terrain_path(path)
            self.edit_status.setText("Terrain settings cleared; using default terrain")
            self.trace_status.setText("Terrain setting cleared")
            update_navigation_ui()

        def on_export_terrain_csv() -> None:
            key = self.terrain_combo.currentText()
            default_name = f"{key.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
            default_dir = Path(__file__).resolve().parent / "exports"
            default_dir.mkdir(parents=True, exist_ok=True)
            default_path = str(default_dir / default_name)

            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Terrain CSV",
                default_path,
                "CSV Files (*.csv)",
            )
            if not path:
                return

            if self._gl.export_current_terrain_csv(path):
                self.edit_status.setText(f"Terrain exported: {Path(path).name}")
            else:
                self.edit_status.setText("Export failed (current terrain is not editable occupancy)")

        def on_import_terrain_csv() -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Terrain CSV",
                str(Path(__file__).resolve().parent),
                "CSV Files (*.csv)",
            )
            if not path:
                return

            if not self._gl.import_terrain_csv(path):
                self.edit_status.setText("Import failed (invalid CSV terrain format)")
                update_navigation_ui()
                return

            if persist_active_terrain():
                self.edit_status.setText(f"Terrain imported and saved: {Path(path).name}")
            else:
                self.edit_status.setText(f"Terrain imported, but save failed: {Path(path).name}")
            update_navigation_ui()

        self.terrain_combo.currentIndexChanged.connect(on_terrain_changed)

        algo_row = QWidget()
        algo_row_layout = QVBoxLayout(algo_row)
        algo_row_layout.setContentsMargins(0, 0, 0, 0)
        algo_row_layout.setSpacing(6)

        algo_select_row = QWidget()
        algo_select_row_layout = QHBoxLayout(algo_select_row)
        algo_select_row_layout.setContentsMargins(0, 0, 0, 0)
        algo_select_row_layout.setSpacing(8)
        algo_select_row_layout.addWidget(algo_label)
        algo_select_row_layout.addWidget(self.algo_combo, stretch=1)

        algo_train_row = QWidget()
        algo_train_row_layout = QHBoxLayout(algo_train_row)
        algo_train_row_layout.setContentsMargins(0, 0, 0, 0)
        algo_train_row_layout.setSpacing(8)
        algo_train_row_layout.addWidget(self.train_algo_button)
        algo_train_row_layout.addWidget(self.algo_status, stretch=1)
        algo_train_row_layout.addWidget(self.algo_progress)

        algo_row_layout.addWidget(algo_select_row)
        algo_row_layout.addWidget(algo_train_row)

        terrain_row = QWidget()
        terrain_row_layout = QVBoxLayout(terrain_row)
        terrain_row_layout.setContentsMargins(0, 0, 0, 0)
        terrain_row_layout.setSpacing(6)

        terrain_select_row = QWidget()
        terrain_select_row_layout = QHBoxLayout(terrain_select_row)
        terrain_select_row_layout.setContentsMargins(0, 0, 0, 0)
        terrain_select_row_layout.setSpacing(8)
        terrain_select_row_layout.addWidget(terrain_label)
        terrain_select_row_layout.addWidget(self.terrain_combo, stretch=1)

        terrain_action_row = QWidget()
        terrain_action_row_layout = QHBoxLayout(terrain_action_row)
        terrain_action_row_layout.setContentsMargins(0, 0, 0, 0)
        terrain_action_row_layout.setSpacing(8)
        terrain_action_row_layout.addWidget(self.clear_terrain_setting_button)
        terrain_action_row_layout.addWidget(self.export_terrain_button)
        terrain_action_row_layout.addWidget(self.import_terrain_button)
        terrain_action_row_layout.addStretch(1)

        terrain_row_layout.addWidget(terrain_select_row)
        terrain_row_layout.addWidget(terrain_action_row)

        terrain_view_separator = QFrame()
        terrain_view_separator.setFrameShape(QFrame.HLine)
        terrain_view_separator.setFrameShadow(QFrame.Sunken)

        self.xyz_label = QLabel("XYZ: --")
        self.xyz_label.setMinimumWidth(140)

        self.axes_checkbox = QCheckBox("Axes")
        self.axes_checkbox.setChecked(True)

        view_row = QWidget()
        view_row_layout = QHBoxLayout(view_row)
        view_row_layout.setContentsMargins(0, 0, 0, 0)
        view_row_layout.setSpacing(8)
        view_row_layout.addWidget(self.xyz_label, stretch=1)
        view_row_layout.addWidget(self.axes_checkbox)

        top_layout.addWidget(algo_row)
        top_layout.addWidget(terrain_view_separator)
        top_layout.addWidget(terrain_row)
        top_layout.addWidget(view_row)

        # --- Terrain editing bar ---
        edit_bar = QWidget()
        edit_layout = QVBoxLayout(edit_bar)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_layout.setSpacing(6)

        edit_title = QLabel("Edit Terrain")
        edit_title.setMinimumWidth(70)

        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Cuboid", "Cylinder"])
        self.shape_combo.setMinimumWidth(90)

        # Cuboid inputs: two diagonal corners (x0,y0,z0) and (x1,y1,z1)
        self.c_x0 = QLineEdit(); self.c_x0.setPlaceholderText("x0")
        self.c_y0 = QLineEdit(); self.c_y0.setPlaceholderText("y0")
        self.c_z0 = QLineEdit(); self.c_z0.setPlaceholderText("z0")
        self.c_x1 = QLineEdit(); self.c_x1.setPlaceholderText("x1")
        self.c_y1 = QLineEdit(); self.c_y1.setPlaceholderText("y1")
        self.c_z1 = QLineEdit(); self.c_z1.setPlaceholderText("z1")

        # Cylinder inputs: base center (cx,cy,cz), radius, height
        self.y_cx = QLineEdit(); self.y_cx.setPlaceholderText("cx")
        self.y_cy = QLineEdit(); self.y_cy.setPlaceholderText("cy")
        self.y_cz = QLineEdit(); self.y_cz.setPlaceholderText("cz")
        self.y_r = QLineEdit(); self.y_r.setPlaceholderText("r")
        self.y_h = QLineEdit(); self.y_h.setPlaceholderText("h")

        for w in [
            self.c_x0, self.c_y0, self.c_z0, self.c_x1, self.c_y1, self.c_z1,
            self.y_cx, self.y_cy, self.y_cz, self.y_r, self.y_h,
        ]:
            w.setFixedWidth(52)

        hint = QLabel("(0-100)")
        hint.setMinimumWidth(50)

        self.add_button = QPushButton("Add")
        self.add_button.setMinimumWidth(70)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setMinimumWidth(70)
        self.undo_button.setEnabled(False)
        self.undo_button.setShortcut("Ctrl+Z")

        self.random_count = QSpinBox()
        self.random_count.setRange(1, 500)
        self.random_count.setValue(10)
        self.random_count.setFixedWidth(72)

        self.random_button = QPushButton("Random Buildings")
        self.random_button.setMinimumWidth(120)

        self.edit_status = QLabel("")
        self.edit_status.setMinimumWidth(120)

        row_shape = QWidget()
        row_shape_layout = QHBoxLayout(row_shape)
        row_shape_layout.setContentsMargins(0, 0, 0, 0)
        row_shape_layout.setSpacing(8)
        row_shape_layout.addWidget(edit_title)
        row_shape_layout.addWidget(self.shape_combo)
        row_shape_layout.addStretch(1)

        self._p0_label = QLabel("P0")
        self._p1_label = QLabel("P1")
        self._c_label = QLabel("C")
        self._r_label = QLabel("r")
        self._h_label = QLabel("h")

        self._cuboid_row_1 = QWidget()
        cuboid_row_1_layout = QHBoxLayout(self._cuboid_row_1)
        cuboid_row_1_layout.setContentsMargins(0, 0, 0, 0)
        cuboid_row_1_layout.setSpacing(6)
        cuboid_row_1_layout.addWidget(self._p0_label)
        cuboid_row_1_layout.addWidget(self.c_x0)
        cuboid_row_1_layout.addWidget(self.c_y0)
        cuboid_row_1_layout.addWidget(self.c_z0)
        cuboid_row_1_layout.addStretch(1)

        self._cuboid_row_2 = QWidget()
        cuboid_row_2_layout = QHBoxLayout(self._cuboid_row_2)
        cuboid_row_2_layout.setContentsMargins(0, 0, 0, 0)
        cuboid_row_2_layout.setSpacing(6)
        cuboid_row_2_layout.addWidget(self._p1_label)
        cuboid_row_2_layout.addWidget(self.c_x1)
        cuboid_row_2_layout.addWidget(self.c_y1)
        cuboid_row_2_layout.addWidget(self.c_z1)
        cuboid_row_2_layout.addStretch(1)

        self._cylinder_row = QWidget()
        cylinder_row_layout = QHBoxLayout(self._cylinder_row)
        cylinder_row_layout.setContentsMargins(0, 0, 0, 0)
        cylinder_row_layout.setSpacing(6)
        cylinder_row_layout.addWidget(self._c_label)
        cylinder_row_layout.addWidget(self.y_cx)
        cylinder_row_layout.addWidget(self.y_cy)
        cylinder_row_layout.addWidget(self.y_cz)
        cylinder_row_layout.addWidget(self._r_label)
        cylinder_row_layout.addWidget(self.y_r)
        cylinder_row_layout.addWidget(self._h_label)
        cylinder_row_layout.addWidget(self.y_h)
        cylinder_row_layout.addStretch(1)

        row_actions = QWidget()
        row_actions_layout = QHBoxLayout(row_actions)
        row_actions_layout.setContentsMargins(0, 0, 0, 0)
        row_actions_layout.setSpacing(8)
        row_actions_layout.addWidget(hint)
        row_actions_layout.addWidget(self.add_button)
        row_actions_layout.addWidget(self.undo_button)
        row_actions_layout.addWidget(self.random_count)
        row_actions_layout.addWidget(self.random_button)
        row_actions_layout.addStretch(1)

        row_status = QWidget()
        row_status_layout = QHBoxLayout(row_status)
        row_status_layout.setContentsMargins(0, 0, 0, 0)
        row_status_layout.setSpacing(8)
        row_status_layout.addWidget(self.edit_status, stretch=1)

        edit_layout.addWidget(row_shape)
        edit_layout.addWidget(self._cuboid_row_1)
        edit_layout.addWidget(self._cuboid_row_2)
        edit_layout.addWidget(self._cylinder_row)
        edit_layout.addWidget(row_actions)
        edit_layout.addWidget(row_status)

        # --- Tracing start/end inputs ---
        trace_bar = QWidget()
        trace_layout = QVBoxLayout(trace_bar)
        trace_layout.setContentsMargins(0, 0, 0, 0)
        trace_layout.setSpacing(6)

        trace_title = QLabel("Trace Points")

        self.sx_edit = QLineEdit(); self.sx_edit.setPlaceholderText("sx")
        self.sy_edit = QLineEdit(); self.sy_edit.setPlaceholderText("sy")
        self.sz_edit = QLineEdit(); self.sz_edit.setPlaceholderText("sz")
        self.ex_edit = QLineEdit(); self.ex_edit.setPlaceholderText("ex")
        self.ey_edit = QLineEdit(); self.ey_edit.setPlaceholderText("ey")
        self.ez_edit = QLineEdit(); self.ez_edit.setPlaceholderText("ez")

        self.sx_edit.setText(str(self._settings.value("trace/sx", "")))
        self.sy_edit.setText(str(self._settings.value("trace/sy", "")))
        self.sz_edit.setText(str(self._settings.value("trace/sz", "")))
        self.ex_edit.setText(str(self._settings.value("trace/ex", "")))
        self.ey_edit.setText(str(self._settings.value("trace/ey", "")))
        self.ez_edit.setText(str(self._settings.value("trace/ez", "")))

        for w in [self.sx_edit, self.sy_edit, self.sz_edit, self.ex_edit, self.ey_edit, self.ez_edit]:
            w.setFixedWidth(58)

        row_trace_title = QWidget()
        row_trace_title_layout = QHBoxLayout(row_trace_title)
        row_trace_title_layout.setContentsMargins(0, 0, 0, 0)
        row_trace_title_layout.setSpacing(8)
        row_trace_title_layout.addWidget(trace_title)
        row_trace_title_layout.addStretch(1)

        row_start = QWidget()
        row_start_layout = QHBoxLayout(row_start)
        row_start_layout.setContentsMargins(0, 0, 0, 0)
        row_start_layout.setSpacing(6)
        row_start_layout.addWidget(QLabel("Start"))
        row_start_layout.addWidget(self.sx_edit)
        row_start_layout.addWidget(self.sy_edit)
        row_start_layout.addWidget(self.sz_edit)
        row_start_layout.addStretch(1)

        row_end = QWidget()
        row_end_layout = QHBoxLayout(row_end)
        row_end_layout.setContentsMargins(0, 0, 0, 0)
        row_end_layout.setSpacing(6)
        row_end_layout.addWidget(QLabel("End"))
        row_end_layout.addWidget(self.ex_edit)
        row_end_layout.addWidget(self.ey_edit)
        row_end_layout.addWidget(self.ez_edit)
        row_end_layout.addStretch(1)

        self.set_trace_button = QPushButton("Set Start/End")
        self.clear_trace_setting_button = QPushButton("Clear Start/End")
        self.clear_trace_setting_button.setMinimumWidth(130)
        self.reset_agent_button = QPushButton("Reset Ball")
        self.navigate_button = QPushButton("Navigate")
        self.navigate_speed = QSpinBox()
        self.navigate_speed.setRange(5, 20)
        self.navigate_speed.setValue(10)
        self.navigate_speed.setSuffix(" u/s")
        self.navigate_speed.setFixedWidth(90)
        self.trace_status = QLabel("")

        self._algo_training_time: dict[str, float] = {
            "A*": 0.0,
            "Q-learning": 0.0,
            "DQN": 0.0,
            "PPO": 0.0,
            "HER": 0.0,
        }
        self._run_counter = 0
        self._metrics_history: list[dict[str, object]] = []

        metrics_bar = QWidget()
        metrics_layout = QVBoxLayout(metrics_bar)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(6)
        metrics_layout.addWidget(QLabel("Metrics"))

        self.metrics_table = QTableWidget(0, 5)
        self.metrics_table.setHorizontalHeaderLabels(["run", "algorithm", "training time", "navigate length", "navigate time"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setMinimumHeight(170)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.metrics_table.setSelectionMode(QTableWidget.SingleSelection)

        hdr = self.metrics_table.horizontalHeader()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)

        def fmt_metric(v: float) -> str:
            return f"{float(v):.3f}"

        def append_metric_log(
            algo_name: str,
            training_time: float,
            navigate_length: float,
            navigate_time: float,
            path: list[tuple[float, float, float]] | None = None,
        ) -> None:
            self._run_counter += 1
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(self._run_counter)))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(algo_name))
            self.metrics_table.setItem(row, 2, QTableWidgetItem(fmt_metric(training_time)))
            self.metrics_table.setItem(row, 3, QTableWidgetItem(fmt_metric(navigate_length)))
            self.metrics_table.setItem(row, 4, QTableWidgetItem(fmt_metric(navigate_time)))
            self._metrics_history.append(
                {
                    "run": self._run_counter,
                    "algorithm": algo_name,
                    "training_time": float(training_time),
                    "navigate_length": float(navigate_length),
                    "navigate_time": float(navigate_time),
                    "path": list(path) if path else [],
                }
            )
            self.metrics_table.scrollToBottom()

        def on_metrics_row_clicked(row: int, _column: int) -> None:
            if row < 0 or row >= len(self._metrics_history):
                self.trace_status.setText("Invalid history row")
                return

            item = self._metrics_history[row]
            path = list(item.get("path", []))
            if not path:
                self.trace_status.setText("No trajectory recorded for this run")
                return

            if self._gl.replay_navigation_path(path):
                algo_name = str(item.get("algorithm", "Unknown"))
                run_no = int(item.get("run", 0))
                self.trace_status.setText(f"Replaying run {run_no} ({algo_name})")
            else:
                self.trace_status.setText("Replay failed for selected run")

        def on_delete_selected_history() -> None:
            row = self.metrics_table.currentRow()
            if row < 0 or row >= self.metrics_table.rowCount():
                self.trace_status.setText("Select a history row to delete")
                return

            self.metrics_table.removeRow(row)
            if 0 <= row < len(self._metrics_history):
                self._metrics_history.pop(row)
            self.trace_status.setText("Selected history deleted")

        def on_clear_history() -> None:
            self.metrics_table.setRowCount(0)
            self._metrics_history.clear()
            self.trace_status.setText("All history cleared")

        metrics_layout.addWidget(self.metrics_table)

        metrics_btn_row = QWidget()
        metrics_btn_layout = QHBoxLayout(metrics_btn_row)
        metrics_btn_layout.setContentsMargins(0, 0, 0, 0)
        metrics_btn_layout.setSpacing(8)
        self.delete_history_button = QPushButton("Delete Selected")
        self.clear_history_button = QPushButton("Clear History")
        metrics_btn_layout.addWidget(self.delete_history_button)
        metrics_btn_layout.addWidget(self.clear_history_button)
        metrics_btn_layout.addStretch(1)
        metrics_layout.addWidget(metrics_btn_row)

        row_trace_actions = QWidget()
        row_trace_actions_layout = QVBoxLayout(row_trace_actions)
        row_trace_actions_layout.setContentsMargins(0, 0, 0, 0)
        row_trace_actions_layout.setSpacing(6)

        row_trace_actions_top = QWidget()
        row_trace_actions_top_layout = QHBoxLayout(row_trace_actions_top)
        row_trace_actions_top_layout.setContentsMargins(0, 0, 0, 0)
        row_trace_actions_top_layout.setSpacing(8)
        row_trace_actions_top_layout.addWidget(self.set_trace_button)
        row_trace_actions_top_layout.addWidget(self.clear_trace_setting_button)
        row_trace_actions_top_layout.addWidget(self.reset_agent_button)
        row_trace_actions_top_layout.addStretch(1)

        row_trace_actions_nav = QWidget()
        row_trace_actions_nav_layout = QHBoxLayout(row_trace_actions_nav)
        row_trace_actions_nav_layout.setContentsMargins(0, 0, 0, 0)
        row_trace_actions_nav_layout.setSpacing(8)
        row_trace_actions_nav_layout.addWidget(self.navigate_button)
        row_trace_actions_nav_layout.addWidget(self.navigate_speed)
        row_trace_actions_nav_layout.addStretch(1)

        row_trace_actions_layout.addWidget(row_trace_actions_top)
        row_trace_actions_layout.addWidget(row_trace_actions_nav)

        row_trace_hint = QWidget()
        row_trace_hint_layout = QHBoxLayout(row_trace_hint)
        row_trace_hint_layout.setContentsMargins(0, 0, 0, 0)
        row_trace_hint_layout.setSpacing(8)
        row_trace_hint_layout.addWidget(QLabel("Move ball: WASD, R/F"))
        row_trace_hint_layout.addWidget(self.trace_status, stretch=1)

        trace_layout.addWidget(row_trace_title)
        trace_layout.addWidget(row_start)
        trace_layout.addWidget(row_end)
        trace_layout.addWidget(row_trace_actions)
        trace_layout.addWidget(row_trace_hint)

        self._gl = RotatingCubeWidget(self, terrain_path=resolved_path)
        self._gl.setMinimumWidth(620)
        self._gl.hover_text_changed.connect(self.xyz_label.setText)
        self._gl.edit_state_changed.connect(self.undo_button.setEnabled)
        self.axes_checkbox.toggled.connect(self._gl.set_show_axes)

        def update_shape_ui() -> None:
            is_cuboid = self.shape_combo.currentText() == "Cuboid"
            for w in [
                self._p0_label,
                self.c_x0,
                self.c_y0,
                self.c_z0,
            ]:
                w.setVisible(is_cuboid)
            self._cuboid_row_1.setVisible(is_cuboid)
            self._cuboid_row_2.setVisible(is_cuboid)
            for w in [
                self._c_label,
                self.y_cx,
                self.y_cy,
                self.y_cz,
                self._r_label,
                self.y_r,
                self._h_label,
                self.y_h,
            ]:
                w.setVisible(not is_cuboid)
            self._cylinder_row.setVisible(not is_cuboid)
            self.edit_status.setText("")

        def parse_f(le: QLineEdit) -> float:
            return float(le.text().strip())

        def on_add() -> None:
            try:
                if self.shape_combo.currentText() == "Cuboid":
                    ok = self._gl.add_cuboid_map(
                        parse_f(self.c_x0), parse_f(self.c_y0), parse_f(self.c_z0),
                        parse_f(self.c_x1), parse_f(self.c_y1), parse_f(self.c_z1),
                    )
                else:
                    ok = self._gl.add_cylinder_map(
                        parse_f(self.y_cx), parse_f(self.y_cy), parse_f(self.y_cz),
                        parse_f(self.y_r), parse_f(self.y_h),
                    )

                if not ok:
                    self.edit_status.setText("Current terrain is not editable (non-HDF5 occupancy)")
                else:
                    if persist_active_terrain():
                        self.edit_status.setText("Added and saved")
                    else:
                        self.edit_status.setText("Added (save failed)")
                update_navigation_ui()
            except Exception:
                self.edit_status.setText("Invalid input")

        def on_undo() -> None:
            if self._gl.undo_last_edit():
                if persist_active_terrain():
                    self.edit_status.setText("Undone and saved")
                else:
                    self.edit_status.setText("Undone (save failed)")
            else:
                self.edit_status.setText("Nothing to undo")
            update_navigation_ui()

        def on_random_buildings() -> None:
            count = int(self.random_count.value())
            added = self._gl.add_random_buildings_map(count)
            if added <= 0:
                self.edit_status.setText("Current terrain is not editable (non-HDF5 occupancy)")
            else:
                if persist_active_terrain():
                    self.edit_status.setText(f"Added {added} random buildings and saved")
                else:
                    self.edit_status.setText(f"Added {added} random buildings (save failed)")
            update_navigation_ui()

        def on_set_trace_points() -> None:
            try:
                sx = float(self.sx_edit.text().strip())
                sy = float(self.sy_edit.text().strip())
                sz = float(self.sz_edit.text().strip())
                ex = float(self.ex_edit.text().strip())
                ey = float(self.ey_edit.text().strip())
                ez = float(self.ez_edit.text().strip())
            except Exception:
                self.trace_status.setText("Invalid start/end input")
                return

            if self._gl.set_trace_points_map(sx, sy, sz, ex, ey, ez):
                self._settings.setValue("trace/sx", f"{sx:g}")
                self._settings.setValue("trace/sy", f"{sy:g}")
                self._settings.setValue("trace/sz", f"{sz:g}")
                self._settings.setValue("trace/ex", f"{ex:g}")
                self._settings.setValue("trace/ey", f"{ey:g}")
                self._settings.setValue("trace/ez", f"{ez:g}")
                self.trace_status.setText("Start/End set; ball spawned at start")
                update_navigation_ui()
            else:
                self.trace_status.setText("Failed to set start/end")

        def on_clear_trace_setting() -> None:
            for k in ["trace/sx", "trace/sy", "trace/sz", "trace/ex", "trace/ey", "trace/ez"]:
                self._settings.remove(k)

            for le in [self.sx_edit, self.sy_edit, self.sz_edit, self.ex_edit, self.ey_edit, self.ez_edit]:
                le.clear()

            self._gl.clear_trace_points()
            self.trace_status.setText("Start/End cleared")
            update_navigation_ui()

        def on_reset_ball() -> None:
            if self._gl.reset_agent_to_start():
                self.trace_status.setText("Ball reset to start")
            else:
                self.trace_status.setText("No start point")

        def update_navigation_ui() -> None:
            algo = self.algo_combo.currentText().strip()
            is_astar = algo == "A*"
            is_rl = algo in {"Q-learning", "DQN", "PPO", "HER"}

            self.train_algo_button.setVisible(is_rl)
            self.algo_progress.setVisible(is_rl)

            trained = self._gl.is_algorithm_trained(algo)
            can_nav = is_astar or trained

            self.navigate_button.setEnabled(can_nav)
            self.navigate_speed.setEnabled(can_nav)

            if is_astar:
                self.algo_status.setText("A* ready")
            elif trained:
                self.algo_status.setText(f"{algo} trained")
            else:
                self.algo_status.setText(f"{algo} not trained")
                self.trace_status.setText("Please train before Navigate")

        def on_train_algorithm() -> None:
            algo = self.algo_combo.currentText().strip()

            self.algo_progress.setValue(0)
            self.algo_status.setText(f"{algo} training...")

            def on_progress(value: int) -> None:
                self.algo_progress.setValue(max(0, min(100, int(value))))
                QApplication.processEvents()

            t0 = perf_counter()
            ok, msg = self._gl.train_navigation_policy(algo, progress_callback=on_progress)
            train_elapsed = perf_counter() - t0

            # A* does not require training.
            self._algo_training_time[algo] = 0.0 if algo == "A*" else float(train_elapsed)

            self.algo_status.setText(msg)
            if ok and self.algo_progress.value() < 100:
                self.algo_progress.setValue(100)
            if ok:
                self.trace_status.setText(f"{algo} training finished")
            else:
                self.trace_status.setText(msg)
            update_navigation_ui()

        def on_navigate() -> None:
            algo = self.algo_combo.currentText().strip()

            self._gl.set_navigation_speed_units_per_sec(float(self.navigate_speed.value()))
            if algo == "A*":
                t0 = perf_counter()
                ok = self._gl.start_astar_navigation()
                nav_elapsed = perf_counter() - t0
                nav_length = float(self._gl.get_last_navigation_length()) if ok else 0.0
                nav_path = self._gl.get_last_navigation_path() if ok else []
                append_metric_log(algo, 0.0, nav_length, float(nav_elapsed), nav_path)
                if ok:
                    pass
                self.trace_status.setText("A* navigation started" if ok else "A* failed: set valid start/end and reachable path")
                return

            if algo == "Q-learning":
                if not self._gl.is_algorithm_trained(algo):
                    self.trace_status.setText("Q-learning is not trained")
                    return
                t0 = perf_counter()
                ok = self._gl.start_q_learning_navigation()
                nav_elapsed = perf_counter() - t0
                nav_length = float(self._gl.get_last_navigation_length()) if ok else 0.0
                nav_path = self._gl.get_last_navigation_path() if ok else []
                append_metric_log(algo, float(self._algo_training_time.get(algo, 0.0)), nav_length, float(nav_elapsed), nav_path)
                if ok:
                    pass
                self.trace_status.setText("Q-learning navigation started" if ok else "Q-learning failed: retrain or reset start/end")
                return

            if algo == "DQN":
                if not self._gl.is_algorithm_trained(algo):
                    self.trace_status.setText("DQN is not trained")
                    return
                t0 = perf_counter()
                ok = self._gl.start_dqn_navigation()
                nav_elapsed = perf_counter() - t0
                nav_length = float(self._gl.get_last_navigation_length()) if ok else 0.0
                nav_path = self._gl.get_last_navigation_path() if ok else []
                append_metric_log(algo, float(self._algo_training_time.get(algo, 0.0)), nav_length, float(nav_elapsed), nav_path)
                self.trace_status.setText("DQN navigation started" if ok else "DQN failed: retrain or reset start/end")
                return

            if algo == "PPO":
                if not self._gl.is_algorithm_trained(algo):
                    self.trace_status.setText("PPO is not trained")
                    return
                t0 = perf_counter()
                ok = self._gl.start_ppo_navigation()
                nav_elapsed = perf_counter() - t0
                nav_length = float(self._gl.get_last_navigation_length()) if ok else 0.0
                nav_path = self._gl.get_last_navigation_path() if ok else []
                append_metric_log(algo, float(self._algo_training_time.get(algo, 0.0)), nav_length, float(nav_elapsed), nav_path)
                self.trace_status.setText("PPO navigation started" if ok else "PPO failed: retrain or reset start/end")
                return

            if algo == "HER":
                if not self._gl.is_algorithm_trained(algo):
                    self.trace_status.setText("HER is not trained")
                    return
                t0 = perf_counter()
                ok = self._gl.start_her_navigation()
                nav_elapsed = perf_counter() - t0
                nav_length = float(self._gl.get_last_navigation_length()) if ok else 0.0
                nav_path = self._gl.get_last_navigation_path() if ok else []
                append_metric_log(algo, float(self._algo_training_time.get(algo, 0.0)), nav_length, float(nav_elapsed), nav_path)
                self.trace_status.setText("HER navigation started" if ok else "HER failed: retrain or reset start/end")
                return

            self.trace_status.setText(f"{algo} navigation is not implemented yet")

        def on_speed_changed(value: int) -> None:
            # Apply speed immediately so running navigation changes speed in real time.
            self._gl.set_navigation_speed_units_per_sec(float(value))

        self.shape_combo.currentIndexChanged.connect(update_shape_ui)
        self.algo_combo.currentIndexChanged.connect(update_navigation_ui)
        self.train_algo_button.clicked.connect(on_train_algorithm)
        self.add_button.clicked.connect(on_add)
        self.undo_button.clicked.connect(on_undo)
        self.random_button.clicked.connect(on_random_buildings)
        self.clear_terrain_setting_button.clicked.connect(on_clear_terrain_setting)
        self.export_terrain_button.clicked.connect(on_export_terrain_csv)
        self.import_terrain_button.clicked.connect(on_import_terrain_csv)
        self.set_trace_button.clicked.connect(on_set_trace_points)
        self.clear_trace_setting_button.clicked.connect(on_clear_trace_setting)
        self.reset_agent_button.clicked.connect(on_reset_ball)
        self.navigate_button.clicked.connect(on_navigate)
        self.navigate_speed.valueChanged.connect(on_speed_changed)
        self.metrics_table.cellClicked.connect(on_metrics_row_clicked)
        self.delete_history_button.clicked.connect(on_delete_selected_history)
        self.clear_history_button.clicked.connect(on_clear_history)
        self.undo_button.setEnabled(self._gl.can_undo_edit())
        update_shape_ui()
        update_navigation_ui()
        on_speed_changed(int(self.navigate_speed.value()))

        # Restore trace points on startup if all six inputs are available and valid.
        try:
            if all(le.text().strip() for le in [self.sx_edit, self.sy_edit, self.sz_edit, self.ex_edit, self.ey_edit, self.ez_edit]):
                on_set_trace_points()
        except Exception:
            pass

        top_edit_separator = QFrame()
        top_edit_separator.setFrameShape(QFrame.HLine)
        top_edit_separator.setFrameShadow(QFrame.Sunken)

        edit_trace_separator = QFrame()
        edit_trace_separator.setFrameShape(QFrame.HLine)
        edit_trace_separator.setFrameShadow(QFrame.Sunken)

        left_layout.addWidget(top_bar)
        left_layout.addWidget(top_edit_separator)
        left_layout.addWidget(edit_bar)
        left_layout.addWidget(edit_trace_separator)
        left_layout.addWidget(trace_bar)
        left_layout.addWidget(metrics_bar)
        left_layout.addStretch(1)

        main_layout.addWidget(left_panel, stretch=0)
        main_layout.addWidget(self._gl, stretch=1)

        self.setCentralWidget(root)

