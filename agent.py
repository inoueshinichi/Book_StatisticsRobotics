import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as anm

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

        

