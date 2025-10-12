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
from kld_mcl import KldMcl


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


if __name__ == "__main__":
    kldmcl1_pattern()


