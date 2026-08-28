"""強化学習のシミュレーション"""
import os
import sys
import math
import inspect

import numpy as np
import matplotlib.pyplot as plt

from map import Map
from obstacle import Puddle
from landmarks import Landmark
from world import PuddleWorld
from goal import Goal
from estimator import (
    KalmanFilterEstimator,
)
from agent import (
    QAgent,
)
from robot import Robot
from sensor import Camera


def qlearning_pattern1():
    time_interval = 0.1
    time_span=400000 # 長時間のアニメーション
    world = PuddleWorld(time_span, time_interval, False)

    # 地図とランドマーク
    m = Map()
    for ln in [(-4,2), (2,-3), (4,4), (-4,-4)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ゴール
    goal = Goal(-3,-3)
    world.append(goal)

    # 水たまり
    world.append(Puddle((-2,0),(0,2), 0.1))
    world.append(Puddle((-0.5,-2),(2.5,1), 0.1))

    # ロボット一台
    init_pose = np.array([3,3,0]).T
    kfe = KalmanFilterEstimator(envmap=m, init_pose=init_pose)
    a = QAgent(time_interval, kfe, goal, disable_init_policy=True)
    r = Robot(pose=init_pose, 
              agent=a, 
              sensor=Camera(envmap=m,
                            distance_bias_rate_stddev=0,
                            direction_bias_stddev=0),
              color='red',
              bias_rate_stds=(0,0))
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


if __name__ == "__main__":
    qlearning_pattern1()