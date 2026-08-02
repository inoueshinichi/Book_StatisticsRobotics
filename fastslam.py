"""
Monte Carlo Localization (MCL)を用いた
パーティクルフィルタによるFAST Slam
"""
import copy
import numpy as np
import math

from mcl import Mcl
from particles import MapParticle

class FastSlam1(Mcl):
    def __init__(self,
                #  envmap,
                 init_pose,
                 particle_num,
                 landmark_num,
                 motion_noise_stds={ 'nn':0.19, 'no':0.001, 'on':0.13, 'oo':0.2 },
                 distance_dev_rate=0.14,
                 direction_dev=0.05):
        # envmap(真のmap)が推定対象になるため、削除
        super().__init__(
                         # envmap, 
                         None,
                         init_pose, 
                         particle_num, 
                         motion_noise_stds, 
                         distance_dev_rate, direction_dev)
        
        # MCLのParticleをMapParticleに上書き
        self.particles = [MapParticle(init_pose, 1.0/particle_num, landmark_num) for i in range(particle_num)]
        self.ml = self.particles[0] # 最尤のパーティクルを新しく作ったパーティクルのリストの先頭にしておく

    def draw(self, ax, elems):
        super().draw(ax, elems)
        self.ml.map.draw(ax, elems)

    # override
    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation,
                                 self.distance_dev_rate,
                                 self.direction_dev) # MCLのobservation_updateからself.mapを削除
        
        # パーティクルの代表と代表値(モード)を決定
        self.set_ml()
            
        # パーティクルのリサンプリング
        self.resampling()

    
class FastSlam2(Mcl):
    def __init__(self,
                 init_pose, 
                 particle_num,
                 landmark_num,
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14,
                 direction_dev=0.05):
        
        super().__init__(
            None,
            init_pose,
            particle_num,
            motion_noise_stds,
            distance_dev_rate,
            direction_dev
        )

        self.particles = [
            MapParticle(init_pose, 1.0/particle_num, landmark_num)
            for i in range(particle_num)
        ]

        self.ml = self.particles[0]

        self.motion_noise_stds = motion_noise_stds

    # override
    def motion_update(self, nu, omega, time, observation):
        # FastSlam2では状態方程式にセンサ値の情報を加えて更新

        # すでに観測されていたランドマークのリストを作成
        not_first_obs = []
        for d in observation:
            # 先頭のパーティクルの地図を使って判断
            if self.particles[0].map.landmarks[d[1]].cov is not None:
                not_first_obs.append(d)
            
        if len(not_first_obs) > 0:
            for p in self.particles:
                p.motion_update2(nu, omega, time,
                                 self.motion_noise_stds,
                                 not_first_obs,
                                 self.distance_dev_rate,
                                 self.direction_dev) # 新しい更新則
        else:
            for p in self.particles:
                p.motion_update(nu, omega,
                                 time,
                                 self.motion_noise_rate_pdf) # 元の更新則
                    
    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.distance_dev_rate, self.direction_dev)
        self.set_ml()
        self.resampling()

    def draw(self, ax, elems):
        super().draw(ax, elems)
        self.ml.map.draw(ax, elems)
    

                