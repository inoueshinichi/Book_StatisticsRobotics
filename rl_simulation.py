"""強化学習のシミュレーション"""
import os
import sys
import math
import inspect

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from map import Map
from obstacle import Puddle
from landmarks import Landmark
from world import PuddleWorld
from goal import Goal
from estimator import (
    KalmanFilterEstimator,
)
from agent import (
    QAgent, SarsaAgent
)
from robot import Robot, WarpRobot
from sensor import Camera


def qlearning_pattern1():
    time_interval = 0.1
    time_span=15
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


def qlearning_pattern2():
    time_interval = 0.1
    time_span=50
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
    a = QAgent(time_interval, 
               kfe,
               goal,
               policy_filename='puddle_ignore_policy_of_value_iteration.txt',
               value_filename='puddle_ignore_values_of_value_iteration.txt',
               disable_init_policy=True,
               )
    r = Robot(pose=init_pose, 
                agent=a, 
                sensor=Camera(envmap=m,
                            distance_bias_rate_stddev=0,
                            direction_bias_stddev=0),
                color='red',
                bias_rate_stds=(0,0))
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def qlearning_pattern3():
    time_interval = 0.1
    time_span = 25
    world = PuddleWorld(time_span, time_interval, debug=False)

    # 地図
    m = Map()
    for ln in [(-4,2), (2,-3), (4,4), (-4,-4)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ゴール
    goal = Goal(-3,-3)
    world.append(goal)

    # 水たまり(障害物)
    world.append(Puddle((-2,0), (0,2), depth=0.1))
    world.append(Puddle((-0.5, -2), (2.5, 1), depth=0.1))

    # ロボット
    init_pose = np.array([3,3,0]).T
    kfe = KalmanFilterEstimator(m, init_pose)
    a = QAgent(time_interval, 
               kfe,
               policy_filename='puddle_ignore_policy_of_value_iteration.txt',
               value_filename='puddle_ignore_values_of_value_iteration.txt',
               disable_init_policy=True)
    r = Robot(init_pose,
              sensor=Camera(
                    envmap=m,
                    distance_bias_rate_stddev=0,
                    distance_noise_rate=0,
              ),
              agent=a,
              color='red',
              bias_rate_stds=(0,0)
        )
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def qlearning_pattern4():
    time_interval = 0.1
    time_span = 25
    world = PuddleWorld(time_span, time_interval, debug=False)

    # 地図
    m = Map()
    for ln in [(-4,2), (2,-3), (4,4), (-4,-4)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ゴール
    goal = Goal(-3,-3)
    world.append(goal)

    # 水たまり(障害物)
    world.append(Puddle((-2,0), (0,2), depth=0.1))
    world.append(Puddle((-0.5, -2), (2.5, 1), depth=0.1))

    # ロボット
    init_pose = np.array([3,3,0]).T
    kfe = KalmanFilterEstimator(m, init_pose)
    a = QAgent(time_interval, 
                kfe,
                disable_init_policy=False)
    r = Robot(init_pose,
                sensor=Camera(
                    envmap=m,
                    distance_bias_rate_stddev=0,
                    distance_noise_rate=0,
                ),
                agent=a,
                color='red',
                bias_rate_stds=(0,0)
        )
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)
    

def qlearning_pattern5():
    time_interval = 0.1
    time_span = 1000
    world = PuddleWorld(time_span, time_interval, debug=False)

    m = Map()
    for ln in [(-4,2),(2,-3),(-4,-4)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    goal = Goal(-3,-3)
    world.append(goal)

    world.append(Puddle((-2,0),(0,2), depth=0.1))
    world.append(Puddle((-0.5,-2),(2.5,1), depth=0.1))

    # ロボットを一台登場させる
    init_pose = np.array([3,3,0]).T
    kfe = KalmanFilterEstimator(m, init_pose)
    a = QAgent(time_interval, 
               kfe,
               policy_filename='puddle_ignore_policy_of_policy_evaluation.txt',
               value_filename='puddle_ignore_values_of_policy_evaluation.txt',
               )
    r = WarpRobot(init_pose,
                  sensor=Camera(m,
                      distance_bias_rate_stddev=0,
                      direction_bias_stddev=0),
                  agent=a,
                  color='red', 
                  bias_rate_stds=(0,0)
                  )
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


    # 方策
    p = np.zeros(a.index_nums[0:2])
    for x in range(a.index_nums[0]):
        for y in range(a.index_nums[1]):
            act = a.ss[(x,y,22)].greedy() # st_idx=22
            p[x,y] = a.actions[act][0] + act.actions[act][1] # uv + uw

    sns.heatmap(np.rot90(p), square=False)
    plt.show()

    # 価値
    v = np.zeros(a.index_nums[0:2])
    for x in range(a.index_nums[0]):
        for y in range(a.index_nums[1]):
            v[x,y] = a.ss[(x,y,18)].max_q()

    sns.heatmap(np.rot90(v), square=False)
    plt.show()
    

def sarsa_pattern1():
    time_interval = 0.1
    time_span = 100
    world = PuddleWorld(time_span, time_interval, debug=False)

    m = Map()
    for ln in [(-4,2),(2,-3),(-4,-4)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    goal = Goal(-3,-3)
    world.append(goal)

    world.append(Puddle((-2,0),(0,2), depth=0.1))
    world.append(Puddle((-0.5,-2),(2.5,1), depth=0.1))

    # ロボットを一台登場させる
    init_pose = np.array([3,3,0]).T
    kfe = KalmanFilterEstimator(m, init_pose)
    a = SarsaAgent(time_interval, 
               kfe,
               policy_filename='puddle_ignore_policy_of_policy_evaluation.txt',
               value_filename='puddle_ignore_values_of_policy_evaluation.txt',
               )
    r = WarpRobot(init_pose,
                  sensor=Camera(m,
                      distance_bias_rate_stddev=0,
                      direction_bias_stddev=0),
                  agent=a,
                  color='red', 
                  bias_rate_stds=(0,0)
                  )
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


    # 方策
    p = np.zeros(a.index_nums[0:2])
    for x in range(a.index_nums[0]):
        for y in range(a.index_nums[1]):
            act = a.ss[(x,y,22)].greedy() # st_idx=22
            p[x,y] = a.actions[act][0] + a.actions[act][1] # uv + uw

    sns.heatmap(np.rot90(p), square=False)
    plt.show()

    # 価値
    v = np.zeros(a.index_nums[0:2])
    for x in range(a.index_nums[0]):
        for y in range(a.index_nums[1]):
            v[x,y] = a.ss[(x,y,18)].max_q()

    sns.heatmap(np.rot90(v), square=False)
    plt.show()

if __name__ == "__main__":
    # qlearning_pattern1()
    # qlearning_pattern2()
    # qlearning_pattern3()
    # qlearning_pattern4()
    qlearning_pattern5()
    sarsa_pattern1()