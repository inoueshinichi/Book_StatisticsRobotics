"""マルコフ決定過程による経路計画問題
    1. 動的計画法 (エージェントが自分の真の位置を知っている場合)
"""
import os
from pathlib import Path
import math
import inspect
import numpy as np
import matplotlib.pyplot as plt

from world import World, PuddleWorld
from map import Map
from landmarks import Landmark
from robot import Robot
from sensor import Camera
from agent import EstimationAgent, PuddleIgnoreAgent, DpPolicyAgent
from goal import Goal
from obstacle import Puddle
from estimator import KalmanFilterEstimator


def mdp_pattern1():
    """環境: ゴールを設定. 
       方策: なし. 固定制御指令(制御速度,制御角速度)
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
    g = Goal(x=-3, y=-3, radius=0.3)
    world.append(g)

    # ロボット
    init_pose = np.array([0, 0, 0]).T # [x,y,theta]
    kfe = KalmanFilterEstimator(envmap=m, init_pose=init_pose) # 推定器
    agent = EstimationAgent(time_interval, 
                            nu=0.2, 
                            omega=10.0/180*math.pi, 
                            estimator=kfe)

    # ロボットは自分の自己位置の真値を知っているものとする(観測誤差 = 0, 状態遷移誤差 = 0)
    robot = Robot(pose=init_pose, 
                  sensor=Camera(envmap=m, 
                                distance_bias_rate_stddev=0, 
                                direction_bias_stddev=0), # 観測誤差 = 0
                  agent=agent, 
                  color='red',
                  bias_rate_stds=(0,0) # 状態遷移誤差 = 0
    )

    world.append(robot)

    world.draw(title=inspect.currentframe().f_code.co_name)


def mdp_pattern2():
    """環境: ゴール + 水たまり(マイナス報酬) 
       方策: なし. 固定制御指令(制御速度,制御角速度)
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
    init_pose = np.array([0, 0, 0]).T # [x,y,theta]
    kfe = KalmanFilterEstimator(envmap=m, init_pose=init_pose) # 推定器
    agent = EstimationAgent(time_interval, 
                            nu=0.2, 
                            omega=10.0/180*math.pi, 
                            estimator=kfe)

    # ロボットは自分の自己位置の真値を知っているものとする(観測誤差 = 0, 状態遷移誤差 = 0)
    robot = Robot(pose=init_pose, 
                  sensor=Camera(m, distance_bias_rate_stddev=0, direction_bias_stddev=0), # 観測誤差 = 0
                  agent=agent, 
                  color='red',
                  bias_rate_stds=(0,0) # 状態遷移誤差 = 0
    )

    world.append(robot)

    world.draw(title=inspect.currentframe().f_code.co_name)


def mdp_pattern3():
    """
    環境: ゴール + 水たまり(マイナス報酬)
    方策: ロボット初期位置からゴールまで一直線に水たまりを無視して突っ切る方策
    強化学習: 1ステップあたりの報酬、収益の獲得
    """

    # シミュレーションパラメータ
    time_interval = 0.1 # [sec]
    time_span = 40 # 40フレーム

    # 環境
    world = PuddleWorld(time_span=time_span, 
                        time_interval=time_interval, 
                        debug=False)

    # 地図
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ゴールの追加
    goal = Goal(x=-3, y=-3, radius=0.3)
    world.append(goal)

    # 水たまり(障害物)の追加
    world.append(Puddle((-2,0),(0,2), depth=0.1))
    world.append(Puddle((-0.5,-2),(2.5,1), depth=0.1))

    # ロボットの初期位置
    init_pose = np.array([2.0, 2.0, math.pi / 2]).T # [x,y,theta]

    # 自己位置推定器
    kfe = KalmanFilterEstimator(envmap=m, init_pose=init_pose) 

    # 固定方策を持つ制御指令エージェント
    agent = PuddleIgnoreAgent(time_interval, 
                              nu=0, 
                              omega=0, 
                              goal=goal,
                              estimator=kfe)

    # シミュレーション外乱設定をしていないロボット
    # 観測誤差=0, 状態遷移誤差=0
    # つまり、ロボットは自己位置の真値をしっている
    robot = Robot(pose=init_pose, 
                  sensor=Camera(envmap=m, 
                                distance_bias_rate_stddev=0, 
                                direction_bias_stddev=0), # 観測誤差 = 0
                  agent=agent, 
                  color='red',
                  bias_rate_stds=(0,0) # 状態遷移誤差 = 0
    )

    world.append(robot)

    world.draw(title=inspect.currentframe().f_code.co_name)

def mdp_pattern4():
    """
    環境：ゴール + 水たまり(マイナス報酬)
    方策：動的計画法により求めた方策. 水たまりがマイナス報酬. ゴールが0.
    強化学習: 方策による制御指令の生成. 1ステップあたりの報酬、収益の獲得
    """
    # シミュレーションパラメータ
    time_interval = 0.1 # [sec]
    time_span = 40 # 40フレーム

    # 環境
    world = PuddleWorld(time_span=time_span, 
                        time_interval=time_interval, 
                        debug=False)

    # 地図
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ゴールの追加
    goal = Goal(x=-3, y=-3, radius=0.3)
    world.append(goal)

    # 水たまり(障害物)の追加
    world.append(Puddle((-2,0),(0,2), depth=0.1))
    world.append(Puddle((-0.5,-2),(2.5,1), depth=0.1))

    ##4台のロボットを動かしてみる##   ##dppolicyagentrun
    init_poses = []
    for p in [[-3, 3, 0], [0.5, 1.5, 0], [3, 3, 0], [2, -1, 0]]:
        init_pose = np.array(p).T
    
        kf = KalmanFilterEstimator(envmap=m, init_pose=init_pose)
        a = DpPolicyAgent(time_interval, kf, goal)
        r = Robot(init_pose, 
                  sensor=Camera(envmap=m, 
                                distance_bias_rate_stddev=0, 
                                direction_bias_stddev=0), 
                  agent=a, 
                  color="red", 
                  bias_rate_stds=(0,0))

        world.append(r)
        
    world.draw(title=inspect.currentframe().f_code.co_name)



if __name__ == "__main__":
    # mdp_pattern1()
    # mdp_pattern2()
    # mdp_pattern3()
    mdp_pattern4()