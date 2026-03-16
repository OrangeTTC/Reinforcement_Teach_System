# Map Module Layout

This project now groups code by folders:

- `algorithm/`: algorithm kernels and algorithm entrypoints.
- `map/`: UI/display/editing map-related code.
- root: startup files.

## Qt Code

- `GUI.py`: app start window and top-level launch flow.
- `map/map_qt_3d_view.py`: Qt main window, controls, and panel wiring.
- `map/map_qt_opengl_widget.py`: Qt OpenGL widget integration and runtime interaction logic.

## 3D Display Code (Engine Driver)

- `map/map_qt_opengl_widget.py`: PyOpenGL-based rendering and interaction.
- `map/map_3d_engine_vpython.py`: VPython scene driver (alternative engine path).

## Terrain Editing

- `map/map_terrain_edit.py`: terrain editing helpers and occupancy editing utilities.

## Terrain Storage and Import

- `map/map_terrain_io.py`: demo map generation and HDF5 storage helpers.
- `map/map_terrain_edit.py`: CSV import/export and HDF5 load/save helpers.

## Entrypoints

- `main.py`: runtime bootstrap and map init.
- `algorithm/algorithm_entrypoints.py`: algorithm train/path entry registry.
