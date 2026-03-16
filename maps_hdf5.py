from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GridSpec:
    nx: int = 100
    ny: int = 100
    nz: int = 100
    voxel_size: float = 1.0


def ensure_demo_maps(workspace_dir: str | Path | None = None) -> None:
    """Create demo HDF5 maps if they don't exist.

    Map format (Scheme 2):
      - HDF5 dataset: /occupancy  (uint8) shape (nx, ny, nz)
      - occupancy[x, y, z] == 1 means blocked / no-fly / occupied.
      - Attributes: voxel_size

    Demo maps:
      - map_flat.h5: ground as a flat surface at height=10 (occupied y<=10)
      - map_mountain.h5: mountain-like ground surface (occupied y<=h(x,z))
    """

    base = Path(workspace_dir) if workspace_dir is not None else Path(__file__).resolve().parent
    spec = GridSpec()

    flat_path = base / "map_flat.h5"
    mountain_path = base / "map_mountain.h5"

    if flat_path.exists() and mountain_path.exists():
        return

    import numpy as np
    import h5py

    nx, ny, nz = spec.nx, spec.ny, spec.nz

    def write_map(path: Path, occ: "np.ndarray") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            dset = f.create_dataset(
                "occupancy",
                data=occ.astype(np.uint8, copy=False),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            dset.attrs["axis_order"] = "x,y,z"
            f.attrs["voxel_size"] = float(spec.voxel_size)

    # Flat: ground at height=10 (inclusive)
    if not flat_path.exists():
        occ = np.zeros((nx, ny, nz), dtype=np.uint8)
        ground_h = 10
        occ[:, : ground_h + 1, :] = 1
        write_map(flat_path, occ)

    # Mountain: height varies with x,z
    if not mountain_path.exists():
        occ = np.zeros((nx, ny, nz), dtype=np.uint8)

        # Normalize coordinates to [-1, 1]
        x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
        z = np.linspace(-1.0, 1.0, nz, dtype=np.float32)
        xx, zz = np.meshgrid(x, z, indexing="xy")

        h = 10.0
        h += 22.0 * np.exp(-((xx - 0.25) ** 2 + (zz + 0.15) ** 2) / (2 * 0.18**2))
        h += 16.0 * np.exp(-((xx + 0.35) ** 2 + (zz - 0.25) ** 2) / (2 * 0.22**2))
        h += 6.0 * np.exp(-((xx + 0.05) ** 2 + (zz + 0.55) ** 2) / (2 * 0.35**2))
        h += 2.0 * np.sin(6 * xx) * np.cos(5 * zz)

        # Clamp to grid range
        h = np.clip(h, 0.0, float(ny - 1))
        h_idx = h.astype(np.int32)

        # Fill occupancy under the surface: occ[x, y<=h(x,z), z] = 1
        # h_idx currently shape (nz, nx) because meshgrid uses (z rows, x cols)
        for zi in range(nz):
            col_h = h_idx[zi]  # length nx
            for xi in range(nx):
                occ[xi, : col_h[xi] + 1, zi] = 1

        write_map(mountain_path, occ)
