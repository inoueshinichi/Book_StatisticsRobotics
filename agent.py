import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as anm

import os
from pathlib import Path

# 制御指令コントローラ
class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega
    

"""Monte Carlo Localization"""
class EstimationAgent(Agent):
    def __init__(self, 
                 time_interval,
                 nu, 
                 omega,
                 estimator):
        super().__init__(nu, omega)
        self.estimator = estimator # 推定器(MCL)
        self.time_interval = time_interval

        # 1ステップ前の状態変数
        self.prev_nu = 0.0
        self.prev_omega = 0.0

    # MCLの状態方程式で更新, 観測方程式で補正
    def decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        return self.nu, self.omega

    
    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)

        # Write ml
        x, y, t = self.estimator.pose
        s = "({:.2f}, {:.2f}, {})".format(x, y, int(t*180/math.pi)%360)
        elems.append(ax.text(x, y+0.1, s, fontsize=8))

        
class FastSlam2Agent(EstimationAgent):
    def __init__(self, time_interval, nu, omega, estimator):
        super().__init__(time_interval, nu, omega, estimator)

    def decision(self, observation=None):
        # センサ値を追加
        self.estimator.motion_update(
            self.prev_nu, 
            self.prev_omega, 
            self.time_interval, 
            observation # 追加
        )

        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        return self.nu, self.omega

    
from robot import IdealRobot

class LoggerAgent(Agent):
    def __init__(self, nu, omega, interval_time, init_pose):
        # 更新時間と初期姿勢を変数に加える
        super().__init__(nu, omega)
        self.interval_time = interval_time
        self.pose = init_pose
        self.step = 0
        current_dir = Path(__file__).resolve().parent
        self.log = open(os.path.join(str(current_dir), "graph_slam_log_1.txt"), "w")

    def decision(self, observation):
        if len(observation) != 0: # ランドマークが観測されていない姿勢は記録しない
            self.log.write("x {} {} {} {}\n".format(self.step, *self.pose))
            for obs in observation:
                # z : step phi Zx, Zy, ZΘ -> カメラとランドマークの相対角度 + (ランドマークの姿勢)
                self.log.write("z {} {} {} {} {}\n".format(self.step, obs[1], *obs[0]))

            self.step += 1
            self.log.flush()

        self.pose = IdealRobot.state_transition(self.nu,
                                                self.omega,
                                                self.interval_time,
                                                self.pose)
        return self.nu, self.omega



    