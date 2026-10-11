"""Physical planning, continuous arcs, acceleration and policy integration."""
from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from algorithm.algorithm_a_star import plan_astar_path_map_xyz
from algorithm.algorithm_entrypoints import PATH_ENTRYPOINTS, TRAIN_ENTRYPOINTS
from algorithm.flight_constraints import (
    FlightConstraints, build_flight_policy_path, build_flight_trajectory,
    map_scale, plan_flight_astar, prepare_flight_grid,
    segment_free, turn_cost,
)


def mapper(value: float, n: int, mode: str) -> int:
    return max(0, min(n-1, round(value * (n-1) / 100)))


class FlightConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.occ = np.zeros((41, 41, 41), dtype=np.uint8)
        self.occ[:, 0, :] = 1
        self.config = FlightConstraints(cruise_speed=2, lateral_accel=2, climb_rate=1)

    def test_parameter_validation_and_radius(self) -> None:
        self.assertEqual(self.config.turn_radius, 72)
        for cfg in (replace(self.config, cruise_speed=13), replace(self.config, lateral_accel=0),
                    replace(self.config, climb_rate=float('nan')), replace(self.config, turn_penalty=-1)):
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_acceleration_braking_and_duration_are_analytic(self) -> None:
        cfg = replace(self.config, max_speed=6, cruise_speed=6, longitudinal_accel=2)
        t = build_flight_trajectory([(0,0,50),(100,0,50)],self.occ,1,cfg)
        self.assertAlmostEqual(t.distance_at_time(1), 1)
        self.assertAlmostEqual(t.speed_at_time(1), 2)
        self.assertAlmostEqual(t.duration, 40/6+3, places=8)
        self.assertAlmostEqual(t.distance_at_time(t.duration), 40)
        self.assertAlmostEqual(t.speed_at_time(t.duration), 0)
        self.assertAlmostEqual(t.distance_at_time(-1), 0)

    def test_non_cubic_grid_physical_conversion_and_arc_length(self) -> None:
        occ = np.zeros((101, 51, 21), dtype=np.uint8)
        np.testing.assert_allclose(map_scale(occ.shape, 2), (2, .4, 1))
        self.assertAlmostEqual(build_flight_trajectory([(0,0,50),(100,0,50)], occ, 2, self.config).length, 200)
        path = [(10,10,50),(50,10,50),(50,60,50)]
        self.occ[12,20,14] = 1  # Block the diagonal without obstructing the two legs.
        t = build_flight_trajectory(path, self.occ, 1, replace(self.config,max_speed=2))
        expected = 16 + 20 - 4 + math.pi
        self.assertAlmostEqual(t.length, expected, places=8)
        arc = next(s for s in t.segments if s.center is not None)
        self.assertAlmostEqual(arc.radius, 2)
        self.assertAlmostEqual(arc.length, 2 * math.pi / 2)
        np.testing.assert_allclose(t.position_map(t.length), path[-1], atol=1e-8)
        # Playback samples from the exact circle, not its chord.
        halfway = arc.position(arc.length / 2)
        self.assertAlmostEqual(np.linalg.norm(halfway - np.asarray(arc.center)), 2)

    def test_narrow_straight_corridor_is_not_dilated(self) -> None:
        occ = np.ones((21,21,21), dtype=np.uint8)
        occ[:,9:12,9:12] = 0
        original=occ.copy()
        t=plan_flight_astar((0,50,50),(100,50,50),occ,1,self.config)
        self.assertAlmostEqual(t.length,20)
        np.testing.assert_array_equal(occ,original)
        np.testing.assert_array_equal(prepare_flight_grid(occ,1,self.config).occupancy,occ)

    def test_climb_speed_adjusts_even_for_vertical_edges(self) -> None:
        cfg=replace(self.config,climb_rate=.4)
        grid=prepare_flight_grid(self.occ,1,cfg)
        self.assertEqual(len(grid.actions),26)
        t=build_flight_trajectory([(50,50,20),(50,50,90)],self.occ,1,cfg)
        self.assertLessEqual(t.speed_profile.speeds.max(),.4+1e-8)
        self.assertAlmostEqual(t.length,28)

    def test_distinct_exact_endpoints_in_same_voxel_are_preserved(self) -> None:
        start, end = (50,50,50), (50.5,50,50)
        t = plan_flight_astar(start,end,self.occ,1,self.config)
        self.assertAlmostEqual(t.length,.2)
        np.testing.assert_allclose(t.position_map(0),start)
        np.testing.assert_allclose(t.position_map(t.length),end)

    def test_diagonal_moves_cannot_cut_edges_or_corners(self) -> None:
        occ = np.zeros((7,7,7),dtype=np.uint8)
        occ[3,3,2] = 1
        cfg = replace(self.config, cruise_speed=.1, lateral_accel=100, climb_rate=1)
        grid = prepare_flight_grid(occ,1,cfg)
        a = grid.actions.index((1,1,1))
        self.assertFalse(grid.valid_moves[(a,2,2,2)])

    def test_exact_segment_collision_and_tangency(self) -> None:
        occ = np.zeros((11,11,11),dtype=np.uint8)
        occ[5,5,5] = 1
        self.assertFalse(segment_free(np.array([0,5,5]), np.array([10,5,5]),occ,1))
        self.assertFalse(segment_free(np.array([0,5.5,5]), np.array([10,5.5,5]),occ,1))
        self.assertTrue(segment_free(np.array([0,6,5]), np.array([10,6,5]),occ,1))

    def test_tight_adjacent_corners_reduce_radius_and_speed(self) -> None:
        occ = np.zeros((41,41,41),dtype=np.uint8)
        occ[12,20,5] = occ[28,20,5] = 1
        path = [(10,10,50),(50,10,50),(50,15,50),(90,15,50)]
        t=build_flight_trajectory(path,occ,1,self.config)
        arcs=[s for s in t.segments if s.center is not None]
        self.assertEqual(len(arcs),2)
        self.assertTrue(all(s.radius < self.config.turn_radius for s in arcs))
        self.assertGreater(t.duration,0)

    def test_astar_obstacle_detour_is_rounded_and_collision_free(self) -> None:
        self.occ[18:23,:27,17:24] = 1
        cfg = replace(self.config, cruise_speed=3, lateral_accel=3, climb_rate=1.5)
        t = plan_flight_astar((10,50,35),(90,50,35),self.occ,1,cfg)
        arcs = [s for s in t.segments if s.center is not None]
        self.assertTrue(arcs)
        for arc in arcs:
            self.assertLessEqual(arc.radius,cfg.turn_radius+1e-8)
        sampled = np.asarray(t.samples_map()) * t.scale
        for a,b in zip(sampled[:-1],sampled[1:]):
            self.assertTrue(segment_free(a,b,self.occ,1))
        self.assertLess(t.length, sum(np.linalg.norm(b-a) for a,b in zip(np.asarray(t.polyline_map[:-1])*t.scale,np.asarray(t.polyline_map[1:])*t.scale)))

    def test_turn_penalties_are_small_and_strengthen_for_nearby_turns(self) -> None:
        u, v = np.array([1,0,0]), np.array([0,0,1])
        self.assertEqual(turn_cost(None,v,0,self.config),0)
        self.assertEqual(turn_cost(u,u,0,self.config),0)
        self.assertAlmostEqual(turn_cost(u,v,100,self.config),.15)
        self.assertAlmostEqual(turn_cost(u,v,0,self.config),.45)

    def test_all_registered_algorithms_use_feasible_graph_and_generate_arcs(self) -> None:
        occ = np.zeros((11,11,11),dtype=np.uint8)
        occ[:,0,:] = 1
        cfg = replace(self.config, cruise_speed=1, lateral_accel=4, climb_rate=.6)
        start, end = (10,10,50),(90,90,50)
        for name, train in TRAIN_ENTRYPOINTS.items():
            with self.subTest(algorithm=name):
                ok,msg,model = train(occ,None,end,mapper,flight_constraints=cfg,voxel_size=1)
                self.assertTrue(ok,msg)
                self.assertEqual(model.flight_constraints,cfg)
                self.assertIsNotNone(model.flight_valid_moves)
                self.assertEqual(model.speed_controller.algorithm,name)
                path = PATH_ENTRYPOINTS[name](model,start,end,mapper)
                self.assertGreaterEqual(len(path),2)
                t = build_flight_trajectory(path,occ,1,cfg,speed_controller=model.speed_controller)
                self.assertGreater(t.length,0)
                np.testing.assert_allclose(t.position_map(t.length),end)

    def test_penalty_changes_policy_choice_with_previous_heading(self) -> None:
        # A tiny synthetic state graph makes scores/ties controllable. Both
        # choices descend distance, so selection must use actual turn history.
        from types import SimpleNamespace
        shape=(5,3,5)
        actions=((1,0,0),(0,0,1))
        valid=np.zeros((2,)+shape,dtype=bool)
        dist=np.full(shape,np.inf)
        q=np.zeros(shape+(2,))
        route=[(0,1,0),(1,1,0),(2,1,0),(2,1,1)]
        for i,node in enumerate(route):
            dist[node]=3-i
        dist[1,1,1]=1.5
        valid[0,0,1,0]=valid[0,1,1,0]=valid[1,1,1,0]=True
        valid[1,2,1,0]=True
        valid[0,1,1,1]=True
        q[1,1,0,1]=.05
        model=SimpleNamespace(flight_constraints=self.config,grid_shape=shape,goal_idx=route[-1],
                              free_mask=np.ones(shape,dtype=bool),distance_map=dist,actions=actions,
                              flight_valid_moves=valid,flight_voxel_size=1,q_table=q)
        path=build_flight_policy_path(model,(0,0,50),(50,25,50))
        self.assertEqual(path[2],(50.,0.,50.))
        model.flight_constraints=replace(self.config,turn_penalty=0,close_turn_penalty=0)
        other=build_flight_policy_path(model,(0,0,50),(50,25,50))
        self.assertEqual(other[2],(25.,25.,50.))


if __name__ == '__main__':
    unittest.main()
