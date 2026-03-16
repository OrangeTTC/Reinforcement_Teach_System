"""
Start window UI module. Provides application entry interactions.

Creator: Orange JYJ
License: Apache-2.0
"""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from map.map_qt_3d_view import ThreeDWindow


class StartWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RL 3D Navigation Demo")
        self.setMinimumSize(520, 320)
        self._child: QMainWindow | None = None

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Reinforcement Learning Navigation Demo")
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        btn_start = QPushButton("Start")
        btn_start.setFixedHeight(48)
        btn_start.clicked.connect(self._open_3d)
        layout.addWidget(btn_start)

        btn_exit = QPushButton("Exit")
        btn_exit.setFixedHeight(48)
        btn_exit.clicked.connect(QApplication.instance().quit)
        layout.addWidget(btn_exit)

        self.setCentralWidget(root)

    def _open_3d(self) -> None:
        # Default: load HDF5 occupancy-grid map.
        self._child = ThreeDWindow(terrain_path="map_flat.h5")
        self._child.show()
        self.close()


def run_gui() -> None:
    """Program entry for the desktop GUI (PySide6)."""

    app = QApplication.instance() or QApplication(sys.argv)
    start = StartWindow()
    start.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    run_gui()
