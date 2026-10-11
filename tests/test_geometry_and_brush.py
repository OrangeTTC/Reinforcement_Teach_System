"""Regression checks for terrain editing."""
from __future__ import annotations

import unittest

import numpy as np

from map.map_terrain_edit import apply_height_brush


class GeometryAndBrushTests(unittest.TestCase):
    def setUp(self) -> None:
        self.occ = np.zeros((11, 11, 11), dtype=np.uint8)
        self.occ[:, 0, :] = 1

    def test_brush_raise_lower_and_locality(self) -> None:
        raised = apply_height_brush(self.occ, (50, 50), 25, 20)
        self.assertEqual(np.flatnonzero(raised[5, :, 5])[-1], 2)
        np.testing.assert_array_equal(raised[0], self.occ[0])
        np.testing.assert_array_equal(self.occ[:, 1:, :], 0)
        lowered = apply_height_brush(raised, (50, 50), 25, -20)
        np.testing.assert_array_equal(lowered[5, :, 5], self.occ[5, :, 5])

    def test_brush_clamps_and_non_square_axes(self) -> None:
        raised = apply_height_brush(self.occ, (50, 50), 25, 1000)
        self.assertEqual(np.flatnonzero(raised[5, :, 5])[-1], 10)
        lowered = apply_height_brush(raised, (50, 50), 25, -1000)
        self.assertEqual(np.flatnonzero(lowered[5, :, 5])[-1], 0)
        occ = np.zeros((11, 7, 21), dtype=np.uint8)
        occ[:, 0, :] = 1
        result = apply_height_brush(occ, (20, 80), 15, 50)
        self.assertGreater(result[2, :, 16].sum(), 1)
        self.assertEqual(result[8, :, 4].sum(), 1)


if __name__ == '__main__':
    unittest.main()
