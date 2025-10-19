"""
Kalman Filter (KF)を用いた自己位置推定
"""
import copy
import math
from pprint import pprint
import numpy as np
import pandas as pd

from world import World
from map import Map
from landmarks import Landmark
from robot import Robot, IdealRobot
from sensor import IdealCamera, Camera
from agent import Agent, EstimationAgent
from kalman_filter import KalmanFilter, GlobalKf

def kl1_pattern():
    time_interval = 0.1
    world = World(40, time_interval, debug=False)

    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    initial_pose = np.array([0,0,0]).T
    kf = KalmanFilter(m, initial_pose)
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, kf)
    r = Robot(initial_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    world.draw()


def kl2_pattern():
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)

    initial_pose = np.array([0,0,0]).T
    kf = KalmanFilter(m, initial_pose)

    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, kf)
    r = Robot(initial_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    kf = KalmanFilter(m, initial_pose)
    linear = EstimationAgent(time_interval, 0.1, 0.0, kf)
    r = Robot(initial_pose, sensor=Camera(m), agent=linear, color="green")
    world.append(r)

    kf = KalmanFilter(m, initial_pose)
    right = EstimationAgent(time_interval, 0.1, -3.0/180*math.pi, kf)
    r = Robot(initial_pose, sensor=Camera(m), agent=right, color="purple")
    world.append(r)

    world.draw()


def kl3_pattern():
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)

    initial_pose = np.array([0,0,0]).T
    kf = KalmanFilter(m, initial_pose)

    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, kf)
    r = Robot(initial_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    kf = KalmanFilter(m, initial_pose)
    linear = EstimationAgent(time_interval, 0.1, 0.0, kf)
    r = Robot(initial_pose, sensor=Camera(m), agent=linear, color="green")
    world.append(r)

    kf = KalmanFilter(m, initial_pose)
    right = EstimationAgent(time_interval, 0.1, -3.0/180*math.pi, kf)
    r = Robot(initial_pose, sensor=Camera(m), agent=right, color="purple")
    world.append(r)

    world.draw()


def kl4_pattern():
    # 大域的自己位置推定
    animation = True
    time_interval = 0.1
    world = World(30, time_interval, debug=not animation)

    m = Map()
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)

    init_pose = np.array([
        np.random.uniform(-5.0,5.0),
        np.random.uniform(-5.0,5.0),
        np.random.uniform(-math.pi, math.pi),
    ]).T

    kf = GlobalKf(m)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, kf)
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()

    # 真の姿勢と推定姿勢を表示
    print(f"GT pose: {r.pose}, Pred pose: {kf.pose}")


def kl5_pattern():
    # ロボット誘拐問題

    animation = True
    time_interval = 0.1
    world = World(30, time_interval, debug=not animation)

    m = Map()
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)

    # ロボットと推定値の初期姿勢が異なる
    init_pose = np.array([
        np.random.uniform(-5.0,5.0),
        np.random.uniform(-5.0,5.0),
        np.random.uniform(-math.pi, math.pi),
    ]).T
    robot_pose = np.array([
        np.random.uniform(-5.0,5.0),
        np.random.uniform(-5.0,5.0),
        np.random.uniform(-math.pi, math.pi),
    ]).T

    kf = KalmanFilter(m, init_pose)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, kf)
    r = Robot(robot_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()

    # 真の姿勢と推定姿勢を表示
    print(f"GT pose: {r.pose}, Pred pose: {kf.pose}")

if __name__ == "__main__":
    # kl1_pattern()
    # kl2_pattern()
    # kl3_pattern()
    # kl4_pattern()
    kl5_pattern()