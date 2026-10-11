"""Adaptive radii, three-dimensional limits and learned velocity actions."""
from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from algorithm.flight_constraints import FlightConstraints, build_flight_trajectory
from algorithm.flight_speed_control import build_speed_profile, segment_speed_limit, speed_features, train_speed_controller


class FlightSpeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.occ = np.zeros((41,41,41),dtype=np.uint8)
        self.config = FlightConstraints(max_speed=6,cruise_speed=6,longitudinal_accel=2,lateral_accel=3,climb_rate=1)

    def assert_limits(self, trajectory) -> None:
        profile=trajectory.speed_profile
        self.assertTrue(np.isfinite(profile.times).all())
        self.assertTrue((np.diff(profile.times)>0).all())
        self.assertLessEqual(profile.speeds.max(),self.config.max_speed+1e-8)
        acceleration=np.diff(profile.speeds**2)/(2*np.diff(profile.distances))
        self.assertLessEqual(np.abs(acceleration).max(),self.config.longitudinal_accel+1e-8)
        cursor=0
        for s in trajectory.segments:
            lo,hi=cursor,cursor+s.length
            nodes=(profile.distances>=lo-1e-8)&(profile.distances<=hi+1e-8)
            v=float(profile.speeds[nodes].max())
            self.assertLessEqual(v,segment_speed_limit(s,self.config)+1e-8)
            if s.center is not None:
                self.assertLessEqual(v*v/s.radius,self.config.lateral_accel+1e-8)
            cursor=hi
        self.assertAlmostEqual(profile.distance_at_time(profile.duration+10),trajectory.length)
        self.assertAlmostEqual(profile.speed_at_time(profile.duration),0)

    def test_three_dimensional_corner_has_continuous_tangents_and_safe_speeds(self) -> None:
        # Block the tempting chord, leaving two unobstructed 3-D legs.
        self.occ[12,20,12]=1
        t=build_flight_trajectory([(10,10,20),(50,10,20),(50,50,80)],self.occ,1,self.config)
        self.assertTrue(any(s.center is not None for s in t.segments))
        for a,b in zip(t.segments[:-1],t.segments[1:]):
            np.testing.assert_allclose(a.end,b.start,atol=1e-7)
            da=a.position(a.length)-a.position(a.length-1e-6)
            db=b.position(1e-6)-b.position(0)
            self.assertGreater(np.dot(da,db)/(np.linalg.norm(da)*np.linalg.norm(db)),.9999)
        self.assert_limits(t)

    def test_short_path_has_triangular_speed_and_correct_time(self) -> None:
        t=build_flight_trajectory([(50,50,50),(50.5,50,50)],self.occ,1,self.config)
        self.assertAlmostEqual(t.duration,2*math.sqrt(.2/self.config.longitudinal_accel))
        self.assert_limits(t)

    def test_all_speed_heads_receive_velocity_and_return_three_actions(self) -> None:
        for name in ('Q-learning','DQN','PPO','HER'):
            with self.subTest(algorithm=name):
                head=train_speed_controller(name,self.config)
                f0=speed_features(.1,6,6,.2,10,self.config)
                f1=speed_features(5,6,1,.2,10,self.config)
                self.assertFalse(np.allclose(head.scores(f0),head.scores(f1)))
                self.assertTrue(np.isfinite(head.scores(f0)).all())
                self.assertEqual(head.scores(f0).shape,(3,))
                t=build_flight_trajectory([(0,0,50),(100,0,50)],self.occ,1,self.config,speed_controller=head)
                self.assertEqual(len(t.speed_profile.actions),len(t.speed_profile.distances)-1)
                self.assert_limits(t)

    def test_speed_actions_change_motion_time_before_safety_projection(self) -> None:
        t=build_flight_trajectory([(0,0,50),(100,0,50)],self.occ,1,self.config)
        class SlowPolicy:
            def choose_action(self,*args):
                return 1 # Hold; the no-stall guard only initiates motion.
        slow=build_speed_profile(t,SlowPolicy())
        self.assertGreater(slow.duration,t.duration*2)
        t.speed_profile=slow
        self.assert_limits(t)

    def test_no_collision_means_direct_AD_shortcut_and_no_artificial_turns(self) -> None:
        t=build_flight_trajectory([(10,10,50),(30,10,50),(30,30,50),(90,90,50)],self.occ,1,self.config)
        self.assertEqual(len(t.segments),1)
        self.assertAlmostEqual(t.length,math.sqrt(2)*32)
        self.assertEqual(len(t.polyline_map),2)


if __name__=='__main__':
    unittest.main()
