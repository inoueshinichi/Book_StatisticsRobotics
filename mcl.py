"""
Monte Carlo Localization(MCL)によるパーティクルフィルタ
"""
import math
import random
import copy
import numpy as np
from scipy.stats import multivariate_normal

from robot import IdealRobot
from sensor import IdealCamera

class Particle:
    def __init__(self, init_pose, weight, noise_mode=False):
        self.pose = init_pose
        self.noise_mode = noise_mode
        self.weight = weight

    def motion_update(self,
                      nu,
                      omega,
                      time,
                      noise_rate_pdf, # 各ステップごとに確率分布が更新される
                      ):
        
        if self.noise_mode:
            # ノイズによる不確かさ(有)
            ns = noise_rate_pdf.rvs() # サンプリング (nn,no,on,oo)
            noised_nu = nu + ns[0] * math.sqrt(abs(nu)/time) + ns[1] * math.sqrt(abs(omega)/time) # 移動量 = 前ステップの移動量 * ノイズ割合
            noised_omega = omega + ns[2] * math.sqrt(abs(nu)/time) + ns[3] * math.sqrt(abs(omega)/time) # 回転量 = 前ステップの回転量 * ノイズ割合
            
            # 状態遷移方程式で現在の状態に更新
            self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)
        else:
            # 不確かさ(無)
            self.pose = IdealRobot.state_transition(nu, omega, time, self.pose)


    def observation_update(self,
                           observation,
                           envmap,
                           distance_dev_rate,
                           direction_dev):
        for d in observation:
            obs_pos = d[0]
            obs_id = d[1]

            # パーティクルの位置と地図からランドマークの距離と方角を算出
            pos_on_map = envmap.landmarks[obs_id].pos
            particle_suggest_pos = IdealCamera.observation_function(self.pose, pos_on_map)

            # 尤度の計算
            distance_dev = distance_dev_rate*particle_suggest_pos[0]
            cov = np.diag(np.array([distance_dev**2, direction_dev**2]))
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos)

class Mcl:
    def __init__(self, 
                 envmap,
                 init_pose, 
                 num,
                 noise_mode,
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14,
                 direction_dev=0.05):
        
        # パーティクルの生成
        self.particles = [
            Particle(init_pose, 1.0/num, noise_mode=noise_mode) for i in range(num)
            ]
        
        self.map = envmap
        self.disetance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

        v = motion_noise_stds
        c = np.diag([v["nn"]**2,v["no"]**2,v["on"]**2,v["oo"]**2]) # 速度と角速度の誤差の分散共分散行列
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)    # 多次元ガウス分布

        self.ml = self.particles[0]
        self.pose = self.ml.pose

    # このメソッドでパーティクルを動かす
    def motion_update(self, 
                      nu, 
                      omega, 
                      time, 
                      ):
        # print(self.motion_noise_rate_pdf.cov) # 共分散行列
        for p in self.particles:
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)

    def draw(self, ax, elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2]) for p in self.particles]
        vys = [math.sin(p.pose[2]) for p in self.particles]
        elems.append(ax.quiver(xs,ys,vxs,vys,color="blue",alpha=0.5))

