"""
KLDサンプリングとMCLを用いた
パーティクルフィルタによる自己位置推定
"""
import copy
import math
import inspect
from pprint import pprint
import numpy as np
import pandas as pd

from world import World
from map import Map
from landmarks import Landmark
from robot import Robot
from sensor import Camera
from agent import EstimationAgent
from estimator import KldMclParticleFilterEstimator, GlobalKldMclParticleFilterEstimator


def kldmcl1_pattern():
    time_interval = 0.1
    world = World(40, time_interval, debug=False)

    # 地図&ランドマーク
    m = Map()
    for ln in [(2,-3), (3,3)]: m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボット
    init_pose = np.array([0,0,0]).T
    kldmclpf = KldMclParticleFilterEstimator(envmap=m, init_pose=init_pose, max_num=1000)
    a = EstimationAgent(time_interval, nu=0.2, omega=10.0/180*math.pi, estimator=kldmclpf)
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def kldmcl2_pattern():
    # 大域的自己位置推定
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    init_pose = np.array([
        np.random.uniform(-4.0,4.0),
        np.random.uniform(-4.0,4.0),
        np.random.uniform(-math.pi, math.pi),
    ]).T

    kldmclpf = GlobalKldMclParticleFilterEstimator(envmap=m, max_num=1000)
    a = EstimationAgent(time_interval, nu=0.2, omega=10.0/180*math.pi, estimator=kldmclpf)
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)

    # 真の姿勢と推定姿勢を表示
    print(f"GT pose: {r.pose}, Pred pose: {kldmclpf.pose}")


def kldmcl3_pattern():
    # 誘拐ロボット問題
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    init_pose = np.array([
        np.random.uniform(-4.0,4.0),
        np.random.uniform(-4.0,4.0),
        np.random.uniform(-math.pi, math.pi),
    ]).T

    robot_pose = np.array([
        np.random.uniform(-4.0,4.0),
        np.random.uniform(-4.0,4.0),
        np.random.uniform(-math.pi, math.pi),
    ]).T

    kldmclpf = KldMclParticleFilterEstimator(envmap=m, 
                                       init_pose=init_pose, 
                                       max_num=10000)

    a = EstimationAgent(time_interval, 
                        nu=0.2, 
                        omega=10.0/180*math.pi,
                        estimator=kldmclpf)
    r = Robot(robot_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)

    # 真の姿勢と推定姿勢を表示
    print(f"GT pose: {r.pose}, Pred pose: {kldmclpf.pose}")

if __name__ == "__main__":
    # kldmcl1_pattern()
    # kldmcl2_pattern()
    kldmcl3_pattern()


