"""マルコフ決定過程による経路計画問題
    1. 動的計画法 (エージェントが自分の真の位置を知っている場合)
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import os

from world import World
from map import Map
from landmarks import Landmark
from robot import Robot
from sensor import Camera
from agent import EstimationAgent
from kalman_filter import KalmanFilter
from goal import Goal
from obstacle import Puddle

def mdp_pattern1():
    """環境にゴールを設定する
    """

    # シミュレーションパラメータ
    time_interval = 0.1 # [sec]
    time_span = 30 # 30フレーム

    # 環境
    world = World(time_span=time_span, time_interval=time_interval, debug=False)

    # 地図
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ゴールの追加
    world.append(Goal(x=-3, y=-3, radius=0.3))

    # 水たまり(障害物)の追加
    world.append(Puddle((-2,0),(0,2), depth=0.1))
    world.append(Puddle((-0.5,-2),(2.5,1), depth=0.1))

    # ロボット
    initial_pose = np.array([0, 0, 0]).T # [x,y,theta]
    kf = KalmanFilter(m, initial_pose) # 推定器
    agent = EstimationAgent(time_interval, nu=0.2, omega=10.0/180*math.pi, estimator=kf)

    # ロボットは自分の自己位置の真値を知っているものとする(観測誤差 = 0, 状態遷移誤差 = 0)
    robot = Robot(pose=initial_pose, 
                  sensor=Camera(m, distance_bias_rate_stddev=0, direction_bias_stddev=0), # 観測誤差 = 0
                  agent=agent, 
                  color='red',
                  bias_rate_stds=(0,0) # 状態遷移誤差 = 0
    )

    world.append(robot)

    world.draw()


if __name__ == "__main__":
    mdp_pattern1()