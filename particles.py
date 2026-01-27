"""パーティクル"""
import math
import copy
import random
import numpy as np
from scipy.stats import multivariate_normal

from robot import IdealRobot
from sensor import IdealCamera
from map import Map
from landmarks import EstimatedLandmark
from kalman_filter import matH, matQ


class Particle:
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight

    # 状態方程式
    def motion_update(self,
                      nu,
                      omega,
                      time,
                      noise_rate_pdf, # 各ステップごとに確率分布が更新される
                      ):
        # ノイズによる不確かさ(有)
        ns = noise_rate_pdf.rvs() # サンプリング (nn,no,on,oo)
        noised_nu = nu + ns[0] * math.sqrt(abs(nu)/time) + ns[1] * math.sqrt(abs(omega)/time) # 移動量 = 前ステップの移動量 * ノイズ割合
        noised_omega = omega + ns[2] * math.sqrt(abs(nu)/time) + ns[3] * math.sqrt(abs(omega)/time) # 回転量 = 前ステップの回転量 * ノイズ割合
            
        # 状態遷移方程式で現在の状態に更新
        self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)
        
    # 観測方程式
    def observation_update(self,
                           observation,
                           envmap,
                           distance_dev_rate,
                           direction_dev,
                           ):
        for d in observation:
            obs_pos = d[0]
            obs_id = d[1]

            # パーティクルの位置と地図からランドマークの距離と方角を算出
            pos_on_map = envmap.landmarks[obs_id].pos # 地図上のランドマーク位置(lx,ly)
            particle_suggest_pos = \
                IdealCamera.observation_function(self.pose, pos_on_map) # カメラからランドマークまでの相対位置(L,φ)

            # 尤度の計算
            distance_dev = distance_dev_rate*particle_suggest_pos[0] # 観測距離が大きいほどノイズは大きいと仮定
            cov = np.diag(np.array([distance_dev**2, direction_dev**2]))
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos) # 尤度(スカラ)を重みにかける

        print(observation)



class MapParticle(Particle):
    def __init__(self, init_pose, weight, landmark_num):
        super().__init__(init_pose, weight)

        self.map = Map() # パーティクルに環境マップを持たせる
        
        for i in range(landmark_num):
            self.map.append_landmark(EstimatedLandmark())

    # ランドマーク位置の初期化
    def init_landmark_estimation(self, landmark, z, distance_dev_rate, direction_dev):
        landmark.pos = z[0] * np.array([
            np.cos(self.pose[2] + z[1]), 
            np.sin(self.pose[2] + z[1])]).T + self.pose[0:2]
        
        # カルマンフィルタのHの右上2x2を取り出し.
        H = matH(self.pose, landmark.pos)[0:2,0:2] # 線形化
        Q = matQ(distance_dev_rate * z[0], direction_dev) # 線形化
        landmark.cov = np.linalg.inv(H.T @ np.linalg.inv(Q) @ H) # ∑ = (H^T Q^-1 H)^-1


    # override
    def observation_update(self, observation, distance_dev_rate, direction_dev):
        for d in observation:
            z = d[0]
            landmark = self.map.landmarks[d[1]]

            if landmark.cov is None:
                # ランドマークが初観測の場合, 初期化
                self.init_landmark_estimation(landmark, z, distance_dev_rate, direction_dev)
            else:
                # ランドマーク更新
                self.observation_update_landmark(landmark, z, distance_dev_rate, direction_dev)

    def observation_update_landmark(self, landmark, z, distance_dev_rate, direction_dev):
        estm_z = IdealCamera.observation_function(self.pose, landmark.pos)
        if estm_z[0] < 0.01: # 推定位置が近すぎると計算がおかしくなるので回避
            return
        
        H = -matH(self.pose, landmark.pos)[0:2, 0:2] # ここは符号の整合性が必要
        Q = matQ(distance_dev_rate * estm_z[0], direction_dev)
        K = landmark.cov @ H.T @ np.linalg.inv(Q + H @ landmark.cov @ H.T) # カルマンゲイン

        # パーティクルの重みの更新
        Q_z = H @ landmark.cov @ H.T + Q
        self.weight *= multivariate_normal(mean=estm_z, cov=Q_z).pdf(z)

        # ランドマークの更新
        landmark.pos = K @ (z - estm_z) + landmark.pos
        landmark.cov = (np.eye(2) - K @ H) @ landmark.cov


