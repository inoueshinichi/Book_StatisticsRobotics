"""
KLDサンプリングとMCLを用いた
パーティクルフィルタによる自己位置推定
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
from mcl import Mcl
from kld_mcl import KldMcl, GlobalKldMcl


def kldmcl1_pattern():
    time_interval = 0.1
    world = World(40, time_interval, debug=False)

    # 地図&ランドマーク
    m = Map()
    for ln in [(2,-3), (3,3)]: m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボット
    initial_pose = np.array([0,0,0]).T
    pf = KldMcl(m, initial_pose, 1000)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, pf)
    r = Robot(initial_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()


def kldmcl2_pattern():
    # 大域的自己位置推定
    animation = True
    time_interval = 0.1
    world = World(30, time_interval, debug=not animation)

    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    init_pose = np.array([
        np.random.uniform(-5.0,5.0),
        np.random.uniform(-5.0,5.0),
        np.random.uniform(-math.pi, math.pi),
    ]).T

    pf = GlobalKldMcl(m, max_num=1000)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, pf)
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()

    # 真の姿勢と推定姿勢を表示
    print(f"GT pose: {r.pose}, Pred pose: {pf.pose}")


def kldmcl3_pattern():
    # 誘拐ロボット問題
    animation = True
    time_interval = 0.1
    world = World(30, time_interval, debug=not animation)

    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

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

    pf = KldMcl(m, init_pose, max_num=10000)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, pf)
    r = Robot(robot_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()

    # 真の姿勢と推定姿勢を表示
    print(f"GT pose: {r.pose}, Pred pose: {pf.pose}")

if __name__ == "__main__":
    # kldmcl1_pattern()
    # kldmcl2_pattern()
    kldmcl3_pattern()


