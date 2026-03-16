from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from qt_3d_view import ThreeDWindow


class TwoDWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("2D View")
        self.setMinimumSize(900, 600)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setAlignment(Qt.AlignCenter)

        label = QLabel("2D mode (placeholder)")
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        label.setFont(font)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        btn = QPushButton("Close")
        btn.setFixedHeight(44)
        btn.setFixedWidth(160)
        btn.clicked.connect(self.close)
        layout.addWidget(btn, alignment=Qt.AlignCenter)

        self.setCentralWidget(root)


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

        btn_2d = QPushButton("2D")
        btn_2d.setFixedHeight(48)
        btn_2d.clicked.connect(self._open_2d)
        layout.addWidget(btn_2d)

        btn_3d = QPushButton("3D")
        btn_3d.setFixedHeight(48)
        btn_3d.clicked.connect(self._open_3d)
        layout.addWidget(btn_3d)

        btn_exit = QPushButton("Exit")
        btn_exit.setFixedHeight(48)
        btn_exit.clicked.connect(QApplication.instance().quit)
        layout.addWidget(btn_exit)

        self.setCentralWidget(root)

    def _open_2d(self) -> None:
        self._child = TwoDWindow()
        self._child.show()
        self.close()

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
