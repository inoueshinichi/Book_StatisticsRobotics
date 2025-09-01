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

    # MCLのmotion_updateを呼ぶ
    def decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        return self.nu, self.omega

    
    def draw(self, ax, elems):
        # elems.append(ax.text(0, 0, "hoge", fontsize=10))
        self.estimator.draw(ax, elems)

