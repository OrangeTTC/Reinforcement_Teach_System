# RL3D Terrain CSV Format

This project now uses a unified CSV format for terrain import/export.

## Format Name

- `format`: `rl3d_terrain_csv_v1`
- `type`: `occupancy_sparse`

## Coordinate Convention

- Occupancy tensor shape is `(nx, ny, nz)`.
- `x`: planar X index, range `[0, nx-1]`
- `y`: height index, range `[0, ny-1]`
- `z`: planar Y index in world/map drawing, range `[0, nz-1]`
- A voxel is occupied when value is `1`.

## CSV Structure

The CSV is row-based with a metadata header and sparse occupancy rows.

1. Metadata rows:
   - `format,rl3d_terrain_csv_v1`
   - `type,occupancy_sparse`
   - `nx,<int>`
   - `ny,<int>`
   - `nz,<int>`
   - `voxel_size,<float>`
2. Column declaration row:
   - `columns,x,y,z`
3. Data rows:
   - `<x>,<y>,<z>` for each occupied voxel.

## Example

```csv
format,rl3d_terrain_csv_v1
type,occupancy_sparse
nx,10
ny,10
nz,10
voxel_size,1.0
columns,x,y,z
0,0,0
0,1,0
1,0,0
5,2,5
5,3,5
```

## Import Rules

- `format` must be `rl3d_terrain_csv_v1`.
- `type` must be `occupancy_sparse`.
- `nx`, `ny`, `nz` must all be >= 2.
- Rows with out-of-range coordinates are ignored.
- Empty lines are ignored.

## Export Rules

- Export writes only occupied voxels (sparse storage).
- Exported CSV always includes metadata rows and `columns,x,y,z`.
- Export uses current editable terrain occupancy in memory.

## UI Behavior

- `Export Terrain CSV`: exports current terrain to this format.
- `Import Terrain CSV`: loads this format and applies it to current terrain.
- After import, terrain is also persisted to the current terrain slot for next startup.
