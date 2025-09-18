"""
カルマンフィルタ
"""
import math

import numpy as np
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

from robot import *


def sigma_ellipse(p, cov, n):
    eig_vals, eig_vec = np.linalg.eig(cov) # 分散共分散行列の固有値と固有ベクトル
    ang = math.atan2(eig_vec[:,0][1], eig_vec[:,0][0]) / math.pi * 180.0 # 長辺の角度
    return Ellipse(p, width=2*n*math.sqrt(eig_vals[0]), height=2*n*math.sqrt(eig_vals[1]), angle=ang, fill=False, color="blue", alpha=0.5)


class KalmanFilter:
    def __init__(self,
                 envmap,    # 環境地図
                 init_pose, # ロボットの初期位置
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 ):
        
        self.belief = multivariate_normal(mean=np.array([0,0,math.pi/4]), cov=np.diag([0.1,0.2,0.01]))
        self.pose = init_pose

    def motion_update(self, nu, omega, time):
        pass
    
    def observation_update(self, observation):
        pass

    def draw(self, ax, elems):
        # XY平面上の誤差3シグマの範囲
        e = sigma_ellipse(self.belief.mean[0:2], self.belief.cov[0:2, 0:2], 3)
        elems.append(ax.add_patch(e))

        # Θ方向の誤差3シグマの範囲
        x,y,c = self.belief.mean
        sigma3 = math.sqrt(self.belief.cov[2,2])*3
        xs = [x + math.cos(c-sigma3), x, x + math.cos(c+sigma3)]
        ys = [x + math.sin(c-sigma3), y, y + math.sin(c+sigma3)]
        elems += ax.plot(xs, ys, color="blue", alpha=0.5)

