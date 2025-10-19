"""
カルマンフィルタ
"""
import math

import numpy as np
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

from robot import *
from sensor import *

def sigma_ellipse(p, cov, n):
    eig_vals, eig_vec = np.linalg.eig(cov) # 分散共分散行列の固有値と固有ベクトル
    ang = math.atan2(eig_vec[:,0][1], eig_vec[:,0][0]) / math.pi * 180.0 # 長辺の角度
    return Ellipse(p, width=2*n*math.sqrt(eig_vals[0]), height=2*n*math.sqrt(eig_vals[1]), angle=ang, fill=False, color="blue", alpha=0.5)

# 更新前(t-1)の推定に対してMtを計算
def matM(nu, omega, time, stds):
    return np.diag([stds["nn"]**2*abs(nu)/time + stds["no"]**2*abs(omega)/time,
                    stds["on"]**2*abs(nu)/time + stds["oo"]**2*abs(omega)/time])

# 更新前(t-1)の推定に対してAtを計算
def matA(nu, omega, time, theta):
    st, ct = math.sin(theta), math.cos(theta)
    stw, ctw = math.sin(theta + omega*time), math.cos(theta + omega*time)
    return np.array([[(stw-st)/omega, -nu/(omega**2)*(stw-st) + nu/omega*time*ctw],
                     [(-ctw+ct)/omega, -nu/(omega**2)*(-ctw+ct) + nu/omega*time*stw],
                     [0, time]])

# 更新前(t-1)の推定に対してFtを計算
def matF(nu, omega, time, theta):
    F = np.diag([1.0, 1.0, 1.0])
    F[0,2] = nu/omega*(math.cos(theta+omega*time)-math.cos(theta))
    F[1,2] = nu/omega*(math.sin(theta+omega*time)-math.sin(theta))
    return F

# 観測方程式を線形化した際の誤差分だけ増加させる行列Ht
def matH(pose, landmark_pos):
    mx, my = landmark_pos
    mux, muy, mut = pose
    q = (mux-mx)**2 + (muy-my)**2
    return np.array([[(mux-mx)/np.sqrt(q), (muy-my)/np.sqrt(q), 0.0],
                     [(my-muy)/q, (mux-mx)/q, -1.0]])

# 観測方程式の共分散行列Qt
def matQ(distance_dev, direction_dev):
    return np.diag(np.array([distance_dev**2, direction_dev**2]))


class KalmanFilter:
    def __init__(self,
                 envmap,    # 環境地図
                 init_pose, # ロボットの初期位置
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14,
                 direction_dev=0.05,
                 ):
        
        # self.belief = multivariate_normal(mean=np.array([0,0,math.pi/4]), 
        #                                   cov=np.diag([0.1,0.2,0.01]))
        self.belief = multivariate_normal(mean=init_pose,
                                          cov=np.diag([1e-10, 1e-10, 1e-10]))
        self.pose = self.belief.mean
        self.motion_noise_stds = motion_noise_stds
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev


    def motion_update(self, nu, omega, time):
        if abs(omega) < 1e-5: omega = 1e-5 # 値が0になるとゼロ割になって計算ができないので微小値を持たせる

        M = matM(nu, omega, time, self.motion_noise_stds)
        A = matA(nu, omega, time, self.belief.mean[2])
        F = matF(nu, omega, time, self.belief.mean[2])
        cov = F @ self.belief.cov @ F.T + A @ M @ A.T
        mean = IdealRobot.state_transition(nu, omega, time, self.belief.mean)
        self.belief = multivariate_normal(mean=mean, cov=cov)
        self.pose = self.belief.mean


    def observation_update(self, observation):
        mean, cov = self.belief.mean, self.belief.cov
        for d in observation:
            z = d[0]
            obs_id = d[1]

            H = matH(mean, self.map.landmarks[obs_id].pos)
            estimated_z = IdealCamera.observation_function(mean, self.map.landmarks[obs_id].pos)
            Q = matQ(estimated_z[0]*self.distance_dev_rate, self.direction_dev)
            K = cov @ H.T @ np.linalg.inv(Q + H @ cov @ H.T)
            mean += K @ (z - estimated_z)
            cov = (np.eye(3) - K @ H) @ cov

        self.belief = multivariate_normal(mean=mean, cov=cov)
        self.pose = self.belief.mean

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


# 大域的自己位置推定
class GlobalKf(KalmanFilter):
    def __init__(self,
                 envmap,
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14,
                 direction_dev=0.05):
        super().__init__(envmap, np.array([0,0,0]).T,
                         motion_noise_stds,
                         distance_dev_rate,
                         direction_dev)
        # 初期値は平均０、分散共分散の値を(大)にしてあやふやにする
        self.belief = multivariate_normal(mean=np.array([0,0,0]),
                                          cov=np.diag([1e+4, 1e+4, 1e+4]))

