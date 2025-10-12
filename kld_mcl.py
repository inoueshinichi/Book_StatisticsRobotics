"""
KLDサンプリング MCLによるパーティクルフィルタ
"""
from mcl import Particle, Mcl
from scipy.stats import chi2 # カイ二乗分布

import copy
import math
import random

import numpy as np

class KldMcl(Mcl):
    def __init__(self,
                 envmap,
                 init_pose,
                 max_num, # 最大生成パーティクル数
                 motion_noise_stds={"nn":0.19, "no":0.001, "on": 0.13, "oo":0.2},
                 distance_dev_rate=0.14,
                 direction_dev=0.05,
                 widths=np.array([0.2, 0.2, math.pi/18]).T,
                 epsilon=0.1,
                 delta=0.01):
        super().__init__(envmap,init_pose,1,motion_noise_stds,distance_dev_rate,direction_dev)
        self.widths = widths # 各空間XYΘのビン幅
        self.max_num = max_num
        self.epsilon = epsilon
        self.delta = delta
        self.binnum = 0 # ビンの数k

    def motion_update(self, nu, omega, time): 
        ws = [e.weight for e in self.particles]
        if sum(ws) < 1e-100: ws = [e + 1e-100 for e in ws] # 重みの和がゼロになるのを回避

        new_particles = [] # 最終的にself.particles
        bins = set() # ビンのインデックス集合
        for i in range(self.max_num):
            chosen_p = random.choices(self.particles, weights=ws) # 戻り値はリスト型
            p = copy.deepcopy(chosen_p[0])
            p.motion_update(nu,omega,time,self.motion_noise_rate_pdf) # パーティクルの移動
            bins.add(tuple(math.floor(e) for e in p.pose/self.widths))#ビンのインデックスをsetに登録（角度を正規化するとより良い）
            new_particles.append(p)

            self.binnum = len(bins) if len(bins) > 1 else 2 #ビンの数が1の場合2にしないと次の行の計算ができない
            y = chi2.ppf(1.0-self.delta, self.binnum-1) # self.binnumのときのχ二乗分布の分位数
            if len(new_particles) > math.ceil(y/(2*self.epsilon)): # 式(7.21)
                break

        self.particles = new_particles # 最終的なパーティクル集合
        for i in range(len(self.particles)): # 重みを正規化
            self.particles[i].weight = 1.0/len(self.particles)

    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        self.set_ml()

    def draw(self, ax, elems):
        super().draw(ax, elems)
        elems.append(ax.text(-4.5,-4.5,"particle:{}, bin:{}".format(len(self.particles),self.binnum), fontsize=10))


    