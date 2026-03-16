"""
Application bootstrap entrypoint. Initializes terrain data and launches GUI.

Creator: Orange JYJ
License: Apache-2.0
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _reexec_into_venv_if_available() -> None:
	"""If a local .venv exists and we're not inside it, re-run using its python.

	This makes `python main.py` work even when dependencies were installed into
	`.venv` (common on Windows).
	"""

	if os.environ.get("VIRTUAL_ENV"):
		return

	workspace_dir = Path(__file__).resolve().parent
	venv_python = workspace_dir / ".venv" / "Scripts" / "python.exe"
	if not venv_python.exists():
		return

	current = Path(sys.executable).resolve()
	try:
		target = venv_python.resolve()
	except OSError:
		target = venv_python
	if current == target:
		return

	result = subprocess.run([str(target), str(Path(__file__).resolve()), *sys.argv[1:]])
	raise SystemExit(result.returncode)


def main() -> int:
	_reexec_into_venv_if_available()

	# Ensure demo HDF5 occupancy-grid maps exist (flat + mountain).
	try:
		from map.map_terrain_io import ensure_demo_maps

		ensure_demo_maps()
	except Exception:
		# If map generation fails, still allow the GUI to launch.
		pass

	from GUI import run_gui

	run_gui()
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

