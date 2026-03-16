from __future__ import annotations

import csv
from pathlib import Path


def derive_heightfield_from_occupancy(occ, voxel_size: float):
    """Derive heights[y,x] from occupancy by taking highest occupied voxel per planar cell."""

    import numpy as np

    occ = np.asarray(occ)
    nx, ny, nz = occ.shape
    if nx < 2 or ny < 2 or nz < 2:
        return None

    mask = occ != 0
    rev = mask[:, ::-1, :]
    idx_from_top = rev.argmax(axis=1)
    any_occ = rev.any(axis=1)
    y_max = (ny - 1 - idx_from_top).astype(np.float32)
    y_max[~any_occ] = 0
    return (y_max.T * float(voxel_size)).astype(float)


def export_occupancy_csv(occupancy, voxel_size: float, output_path: str | Path) -> bool:
    """Export occupancy tensor to unified sparse CSV format."""

    try:
        import numpy as np

        occ = np.asarray(occupancy, dtype=np.uint8)
        if occ.ndim != 3:
            return False

        nx, ny, nz = occ.shape
        occupied = np.argwhere(occ != 0)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["format", "rl3d_terrain_csv_v1"])
            writer.writerow(["type", "occupancy_sparse"])
            writer.writerow(["nx", str(nx)])
            writer.writerow(["ny", str(ny)])
            writer.writerow(["nz", str(nz)])
            writer.writerow(["voxel_size", f"{float(voxel_size):g}"])
            writer.writerow(["columns", "x", "y", "z"])
            for x, y, z in occupied:
                writer.writerow([int(x), int(y), int(z)])
        return True
    except Exception:
        return False


def load_occupancy_csv(csv_path: str | Path):
    """Load occupancy tensor from unified sparse CSV format.

    Returns (occupancy, voxel_size) on success, otherwise None.
    """

    try:
        import numpy as np

        p = Path(csv_path)
        with p.open("r", newline="", encoding="utf-8") as f:
            rows = []
            for row in csv.reader(f):
                if not row:
                    continue
                trimmed = [c.strip() for c in row]
                if not any(trimmed):
                    continue
                rows.append(trimmed)

        if not rows:
            return None

        if len(rows[0]) < 2 or rows[0][0].lower() != "format" or rows[0][1] != "rl3d_terrain_csv_v1":
            return None

        meta: dict[str, str] = {}
        data_start = None
        for i, row in enumerate(rows):
            key = row[0].lower()
            if key == "columns":
                data_start = i + 1
                break
            if len(row) < 2:
                return None
            meta[key] = row[1]

        if data_start is None:
            return None

        if meta.get("type", "") != "occupancy_sparse":
            return None

        nx = int(meta["nx"])
        ny = int(meta["ny"])
        nz = int(meta["nz"])
        voxel_size = float(meta.get("voxel_size", "1.0"))

        if nx < 2 or ny < 2 or nz < 2:
            return None

        occ = np.zeros((nx, ny, nz), dtype=np.uint8)
        for row in rows[data_start:]:
            if len(row) < 3:
                continue
            x = int(float(row[0]))
            y = int(float(row[1]))
            z = int(float(row[2]))
            if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                occ[x, y, z] = 1

        return (occ, voxel_size)
    except Exception:
        return None


def save_occupancy_hdf5(occupancy, voxel_size: float, output_path: str | Path) -> bool:
    """Save occupancy tensor to HDF5 file using dataset /occupancy."""

    try:
        import numpy as np
        import h5py

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(target, "w") as f:
            dset = f.create_dataset(
                "occupancy",
                data=np.asarray(occupancy, dtype=np.uint8),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            dset.attrs["axis_order"] = "x,y,z"
            f.attrs["voxel_size"] = float(voxel_size)
        return True
    except Exception:
        return False


def load_terrain_hdf5(h5_path: str | Path):
    """Load terrain from HDF5.

    Returns (heights, voxel_size, grid_ny, occupancy) or (None, _, _, _).
    """

    try:
        import numpy as np
        import h5py

        p = Path(h5_path)
        with h5py.File(p, "r") as f:
            voxel_size = float(f.attrs.get("voxel_size", 1.0))

            if "heightfield" in f:
                hf = np.asarray(f["heightfield"][...], dtype=float)
                if hf.ndim != 2 or hf.shape[0] < 2 or hf.shape[1] < 2:
                    return (None, voxel_size, None, None)
                return (hf, voxel_size, None, None)

            if "occupancy" not in f:
                return (None, voxel_size, None, None)

            occ = np.asarray(f["occupancy"][...])
            if occ.ndim != 3:
                return (None, voxel_size, None, None)

            nx, ny, nz = occ.shape
            if nx < 2 or ny < 2 or nz < 2:
                return (None, voxel_size, None, None)

            heights = derive_heightfield_from_occupancy(occ, voxel_size)
            if heights is None:
                return (None, voxel_size, None, None)

            return (heights, voxel_size, int(ny), occ.astype(np.uint8, copy=False))
    except Exception:
        return (None, 1.0, None, None)
