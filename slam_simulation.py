"""
Monte Carlo Localization (MCL)を用いた
パーティクルフィルタによるSlamシミュレーション
"""
import numpy as np
import math

from world import World
from map import Map
from landmarks import Landmark
from robot import Robot
from sensor import Camera
from agent import EstimationAgent
from fastslam import FastSlam


def slam1_pattern():
    time_interval = 0.1
    world = World(time_span=30, time_interval=time_interval, debug=False)

    # 真の地図を作成
    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボット作成
    init_pose = np.array([0,0,0]).T
    pf = FastSlam(m, init_pose, particle_num=100, landmark_num=len(m.landmarks))
    a = EstimationAgent(time_interval, 
                        nu=0.2, 
                        omega=10.0/180 * math.pi, 
                        estimator=pf)
    r = Robot(init_pose, sensor=Camera(m), agent=a, color='red')
    world.append(r)

    world.draw()


def slam2_pattern():
    time_interval = 0.1
    
    world = World(time_span=30, time_interval=time_interval, debug=False)

    ##真の地図を作成##
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]: m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボット作成
    init_pose = np.array([0,0,0]).T
    pf = FastSlam(init_pose, particle_num=100, landmark_num=len(m.landmarks))
    a = EstimationAgent(time_interval, 
                        nu=0.2, 
                        omega=10.0/180 * math.pi, 
                        estimator=pf)
    r = Robot(init_pose, sensor=Camera(m), agent=a, color='red')
    world.append(r)

    world.draw()


if __name__ == "__main__":
    # slam1_pattern()
    slam2_pattern()
