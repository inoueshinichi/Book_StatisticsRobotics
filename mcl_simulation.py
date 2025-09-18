"""
Monte Carlo Localization (MCL)を用いた
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


def mcl1_pattern():
    world = World(60, 0.1)

    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    initial_pose = np.array([2,2,math.pi/6]).T
    circling = EstimationAgent(nu=0.2, omega=10.0/180*math.pi)
    r = Robot(initial_pose, sensor=Camera(m), agent=circling, expected_kidnap_time=None)
    world.append(r)

    world.draw()


def mcl2_pattern():
    world = World(30, 0.1)

    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    initial_pose = np.array([2,2,math.pi/6]).T
    estimator = Mcl(initial_pose, 100) # パーティクルフィルタを作る
    circling = EstimationAgent(nu=0.4, omega=20.0/180*math.pi, estimator=estimator) 
    r = Robot(initial_pose, sensor=Camera(m), agent=circling, expected_kidnap_time=None)
    world.append(r)

    world.draw()


def mcl3_pattern():
    initial_pose = np.array([0,0,0]).T
    estimator = Mcl(initial_pose, 100, motion_noise_stds={"nn":0.01, "no":0.02, "on":0.03, "oo":0.04})
    a = EstimationAgent(time_interval=0.1, 
                        nu=0.2,
                        omega=10.0/180*math.pi,
                        estimator=estimator)
    estimator.motion_update(0.2, 10.0/180*math.pi, time=0.1)
    for p in estimator.particles:
        print(p.pose)


def mcl4_pattern():
    time_interval = 0.1 # 0.1[s]
    world = World(60, time_interval)

    initial_pose = np.array([0,0,0]).T
    estimator = Mcl(initial_pose, 100, motion_noise_stds={"nn":0.01, "no":0.02, "on":0.03, "oo":0.04})
    circling = EstimationAgent(time_interval=time_interval,
                               nu=0.2,
                               omega=10.0/180*math.pi,
                               estimator=estimator)
    r = Robot(initial_pose, sensor=None, agent=circling, color="red")
    world.append(r)

    world.draw()


def mcl5_pattern(motion_noise_stds):
    time_interval = 0.1 # 0.1[s]
    world = World(60, time_interval)

    initial_pose = np.array([0,0,0]).T
    estimator = Mcl(initial_pose, 100, motion_noise_stds=motion_noise_stds)
    circling = EstimationAgent(time_interval=time_interval,
                               nu=0.2,
                               omega=10.0/180*math.pi,
                               estimator=estimator)
    r = Robot(initial_pose, sensor=None, agent=circling, expected_kidnap_time=None, color="red")
    world.append(r)

    world.draw()




def motion_test_forward():
    # ロボットの動きをブラックボックスと見立てて,
    # 繰り返し実験からロボットの雑音パラメータを決定する作業

    world = World(40, 0.1)

    initial_pose = np.array([0,0,0]).T
    robots = []
    r = Robot(initial_pose, sensor=None, agent=Agent(0.1, 0.0))

    for i in range(100):
        copy_r = copy.copy(r)
        copy_r.distance_until_noise = copy_r.noise_pdf.rvs() # 最初に雑音が発生するタイミングを変える
        world.append(copy_r)
        robots.append(copy_r)   

    world.draw() 


def motion_test_forward_bias():
    world = World(40.0, 0.1)
    initial_pose = np.array([0,0,0]).T
    robots = []

    for i in range(100):
        r = Robot(initial_pose, sensor=None, agent=Agent(0.1, 0.0)) # ここで生成されるロボットは異なるバイアスを持つ
        world.append(r)
        robots.append(r)

    world.draw()

    poses = pd.DataFrame([[math.sqrt(r.pose[0]**2 + r.pose[1]**2), r.pose[2]] for r in robots],
                         columns=['r', 'theta'])
    print(poses.transpose())

    print(poses["r"].var()) 
    print(poses["r"].mean())
    print(math.sqrt(poses["r"].var()/poses["r"].mean()))


def motion_test_rot_bias():
    world = World(40.0, 0.1)

    initial_pose = np.array([0,0,0]).T
    robots = []

    for i in range(100):
        r = Robot(initial_pose, sensor=None, agent=Agent(0,0.1))
        world.append(r)
        robots.append(r)
    
    world.draw()

    poses = pd.DataFrame([[math.sqrt(r.pose[0]**2 + r.pose[1]**2), r.pose[2]] 
                          for r in robots], columns=['r', 'theta'])
    poses.transpose()

    print(poses["theta"].var())
    print(poses["theta"].mean())
    math.sqrt(poses["theta"].var()/poses["theta"].mean())


def mcl6_pattern():
    time_interval = 0.1 # 0.1[s]
    world = World(40, time_interval)

    initial_pose = np.array([0,0,0]).T

    estimator = Mcl(initial_pose, 
                    100, 
                    motion_noise_stds={"nn":0.001, "no":0.001, "on":0.13, "oo":0.001})
    a = EstimationAgent(time_interval=time_interval,
                        nu=0.1,
                        omega=0.0,
                        estimator=estimator)

    r = Robot(pose=initial_pose, sensor=None, agent=a, color="red")
    world.append(r)

    world.draw()


def mcl7_pattern():
    time_interval = 0.1
    world = World(40.0, time_interval)

    initial_pose = np.array([0,0,0]).T

    estimator = Mcl(initial_pose,
                    100,
                    motion_noise_stds={"nn":0.001, "no":0.001, "on":0.13, "oo":0.001})
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    r = Robot(initial_pose, sensor=None, agent=circling, color="red" )
    world.append(r)

    world.draw()


def mcl8_pattern():
    time_interval = 0.1

    world = World(40.0, time_interval)

    for i in range(100):
        r = Robot(np.array([0,0,0]).T, sensor=None, agent=Agent(0.2, 10.0/180*math.pi), color="grey")
        world.append(r)
    
    world.draw()


def mcl9_pattern():
    time_interval = 0.1
    world = World(40.0, time_interval, debug=False)

    # 地図を生成して3つのランドマークを追加
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボットを作る
    initial_pose = np.array([0,0,0]).T
    estimator = Mcl(initial_pose, 100)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    r = Robot(initial_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()


def mcl10_pattern():
    time_interval = 0.1
    world = World(40.0, time_interval, debug=True)

    # 地図を生成して3つのランドマークを追加
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボットを作る
    initial_pose = np.array([0,0,0]).T
    estimator = Mcl(initial_pose, 100)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    r = Robot(initial_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()


def sensor_experiment():
    m = Map()
    m.append_landmark(Landmark(1,0))

    distance = []
    direction = []

    for i in range(100):
        c = Camera(m) # バイアスの影響も考慮するために毎回カメラを新規作成
        d = c.data(np.array([0,0,0]).T) # カメラ位置
        if len(d) > 0:
            distance.append(d[0][0][0])
            direction.append(d[0][0][1])
    
    df = pd.DataFrame()
    df["distance"] = distance
    df["direction"] = direction

    pprint(df)
    print()
    print(df.std()) # 標準偏差(距離,角度)
    print(df.mean()) # 平均(距離,角度)


def mcl11_pattern():
    time_interval = 0.1
    world = World(40, time_interval, debug=False)

    # 地図 & ランドマーク
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボット
    initial_pose = np.array([0,0,0]).T
    estimator = Mcl(m, initial_pose, 100) # EstimatorにMapを渡す
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    r = Robot(initial_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()


def mcl12_pattern():
    time_interval = 0.1
    world = World(40, time_interval, debug=False)

    # 地図&ランドマーク
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボット
    initial_pose = np.array([0,0,0]).T
    estimator = Mcl(m, initial_pose, 100)
    a = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    r = Robot(initial_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw()


if __name__ == "__main__":
    # mcl1_pattern()
    # mcl2_pattern()
    # mcl3_pattern()
    # mcl4_pattern() # パーティクルの分布(エージェントの推定分布)が狭すぎて実機の姿勢が乖離した状態
    # mcl5_pattern({"nn":0.01, "no":0.02, "on":0.03, "oo":0.04}) # パーティクルの分布(エージェントの推定分布)が狭すぎて実機の姿勢が乖離した状態
    # mcl5_pattern({"nn":1, "no":2, "on":3, "oo":4}) # 分布が広すぎて推定できていない状態
    # mcl5_pattern({"nn":0.01, "no":0.01, "on":0.01, "oo": 0.3}) # 回転にノイズが大きい
    # mcl5_pattern({"nn":0.1, "no":0.01, "on":0.01, "oo":0.01}) # 移動にノイズが大きい
    # mcl5_pattern({"nn":0.1, "no":0.01, "on":0.01, "oo":0.001}) # 回転にノイズが小さい
    # mcl5_pattern({"nn":0.02, "no":0.01, "on":0.01, "oo":0.1}) # 移動にノイズが小さい
    # motion_test_forward()
    # motion_test_forward_bias()
    # motion_test_rot_bias()
    # mcl6_pattern()
    # mcl7_pattern()
    # mcl8_pattern()
    # mcl9_pattern()
    # mcl10_pattern()
    # sensor_experiment()
    # mcl11_pattern()
    mcl12_pattern()

