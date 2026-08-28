"""
Kalman Filter (KF)を用いた自己位置推定
"""
import copy
import math
from pprint import pprint
import numpy as np
import pandas as pd
import inspect

from world import World
from map import Map
from landmarks import Landmark
from robot import Robot, IdealRobot
from sensor import IdealCamera, Camera
from agent import Agent, EstimationAgent
from estimator import KalmanFilterEstimator, GlobalKalmanFilterEstimator

def kf1_pattern():
    """KFによるロボットの自己位置推定
    ロボットの軌跡: 円軌道
    観測ランドマーク: 3つ
    """
    time_interval = 0.1
    world = World(40, time_interval, debug=False)

    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    init_pose = np.array([0,0,0]).T
    kfe = KalmanFilterEstimator(envmap=m, init_pose=init_pose)
    circling = EstimationAgent(time_interval, 
                               nu=0.2, 
                               omega=10.0/180*math.pi, 
                               estimator=kfe)
    r = Robot(init_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def kf2_pattern():
    """様々なロボットの軌道に対するKF自己位置推定
    ロボット1: 円軌道
    ロボット2: 直線軌道
    ロボット3: 右回転軌道
    観測ランドマーク: 3つ
    """
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)

    init_pose = np.array([0,0,0]).T
    kfe = KalmanFilterEstimator(envmap=m, init_pose=init_pose)

    circling = EstimationAgent(time_interval, 
                               nu=0.2, 
                               omega=10.0/180*math.pi, 
                               estimator=kfe)
    r = Robot(init_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    kfe = KalmanFilterEstimator(envmap=m, init_pose=init_pose)
    linear = EstimationAgent(time_interval, 
                             nu=0.1, 
                             omega=0.0, 
                             estimator=kfe)
    r = Robot(init_pose, sensor=Camera(m), agent=linear, color="green")
    world.append(r)

    kfe = KalmanFilterEstimator(envmap=m, init_pose=init_pose)
    right = EstimationAgent(time_interval, 
                            nu=0.1, 
                            omega=-3.0/180*math.pi, 
                            estimator=kfe)
    r = Robot(init_pose, sensor=Camera(m), agent=right, color="purple")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def kf3_pattern():
    """様々なロボットの軌道に対するKF自己位置推定
    ロボットの制御指令がそれぞれ異なる
    ロボット1: 円軌道
    ロボット2: 円軌道
    ロボット3: 円軌道
    観測ランドマーク: 3つ
    """
    time_interval = 0.1
    world = World(30, time_interval, debug=False)

    m = Map()
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)

    init_pose = np.array([0,0,0]).T
    kfe = KalmanFilterEstimator(envmap=m, init_pose=init_pose)

    circling = EstimationAgent(time_interval, 
                               nu=0.2, 
                               omega=10.0/180*math.pi, 
                               estimator=kfe)
    r = Robot(init_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    kfe = KalmanFilterEstimator(envmap=m, init_pose=init_pose)
    linear = EstimationAgent(time_interval, 
                             nu=0.1, 
                             omega=0.0, 
                             estimator=kfe)
    r = Robot(init_pose, sensor=Camera(m), agent=linear, color="green")
    world.append(r)

    kfe = KalmanFilterEstimator(envmap=m, init_pose=init_pose)
    right = EstimationAgent(time_interval, 
                            nu=0.1, 
                            omega=-3.0/180*math.pi, 
                            estimator=kfe)
    r = Robot(init_pose, sensor=Camera(m), agent=right, color="purple")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def kf4_pattern():
    """カルマンフィルタによる大域的自己位置推定
    """
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

    gkfe = GlobalKalmanFilterEstimator(envmap=m)
    a = EstimationAgent(time_interval, 
                        nu=0.2, 
                        omega=10.0/180*math.pi, estimator=gkfe)
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)

    # 真の姿勢と推定姿勢を表示
    print(f"GT pose: {r.pose}, Pred pose: {gkfe.pose}")


def kf5_pattern():
    """カルマンフィルタによる自己位置推定に対するロボット誘拐問題
    """

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

    kfe = KalmanFilterEstimator(envmap=m, init_pose=init_pose)
    a = EstimationAgent(time_interval, 
                        nu=0.2, 
                        omega=10.0/180*math.pi, 
                        estimator=kfe)
    r = Robot(robot_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)

    # 真の姿勢と推定姿勢を表示
    print(f"GT pose: {r.pose}, Pred pose: {kfe.pose}")

if __name__ == "__main__":
    # kf1_pattern()
    # kf2_pattern()
    # kf3_pattern()
    # kf4_pattern()
    kf5_pattern()