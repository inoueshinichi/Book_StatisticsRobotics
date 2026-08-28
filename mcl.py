"""
パーティクルとMonte Carlo Localization(MCL)を用いたパーティクルフィルタ
"""
import math
import random
import copy
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from scipy.stats import multivariate_normal

from particles import Particle

from map import Map

# 単純なMCL
class Mcl:
    def __init__(self, 
                 envmap: Map,
                 init_pose: np.ndarray, 
                 num: int,
                 motion_noise_stds: Dict[str, float] = {
                     "nn":0.19, "no":0.001, "on":0.13, "oo":0.2
                 },
                 distance_dev_rate: float = 0.14,
                 direction_dev: float = 0.05,
                 ):
        
        # パーティクルの生成
        self.particles = [Particle(init_pose, 1.0/num) for i in range(num)]
        
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

        v = motion_noise_stds
        c = np.diag([v["nn"]**2,v["no"]**2,v["on"]**2,v["oo"]**2]) # 速度と角速度の誤差の分散共分散行列
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)    # 多次元ガウス分布

        # 最尤なパーティクル属性
        self.ml = self.particles[0]
        self.pose = self.ml.pose

    # 最尤パーティクルを選ぶ
    def set_ml(self):
        i = np.argmax([p.weight for p in self.particles])
        self.ml = self.particles[i]
        self.pose = self.ml.pose

    # このメソッドでパーティクルを動かす
    def motion_update(self, nu: float, omega: float, time: float, 
                      ):
        # print(self.motion_noise_rate_pdf.cov) # 共分散行列
        for p in self.particles:
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)

    # パーティクルが地図のランドマークを観測する
    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, 
                                 self.map, 
                                 self.distance_dev_rate,
                                 self.direction_dev)
            
        # パーティクルの代表と代表値(モード)を決定
        self.set_ml()
            
        # パーティクルのリサンプリング
        self.resampling()
            
    # リサンプリング
    # def resampling(self):
    #     ws = [e.weight for e in self.particles] # 重みリスト

    #     # 重みの和がゼロに丸め込まれるとエラーになるので小さな値を足す
    #     if sum(ws) < 1e-100:
    #         ws = [e + 1e-100 for e in ws]

    #     # wsの要素に比例した確率でパーティクルをnum個選択
    #     ps = random.choices(self.particles, weights=ws, k=len(self.particles))

    #     # 選んだリストからパーティクルを取り出し、重みを均一に正規化
    #     self.particles = [copy.deepcopy(e) for e in ps]
    #     for p in self.particles:
    #         p.weight = 1.0/len(self.particles)


    # 系統リサンプリング
    def resampling(self):
        ws = np.cumsum([e.weight for e in self.particles]) # 重みの累積値（最後の要素が重みの合計になる）

        # 重みの和がゼロに丸め込まれるとエラーになるので小さな値を足す
        if ws[-1] < 1e-100: # ws[-1]は重みの和
            ws = [e + 1e-100 for e in ws]

        step = ws[-1] / len(self.particles) # 正規化されていない場合はステップが「重みの合計値/N」になる
        r = np.random.uniform(0.0, step)
        cur_pos = 0
        ps = [] # 抽出するパーティクルのリスト

        while (len(ps) < len(self.particles)):
            if r < ws[cur_pos]:
                ps.append(self.particles[cur_pos]) #もしかしたらcur_posがはみ出るかもしれませんが例外処理は割愛で
                r += step
            else:
                cur_pos += 1

        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles:
            p.weight = 1.0 / len(self.particles)

    def draw(self, ax, elems):
        # print(f"mcl draw")
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        vys = [math.sin(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        blue_quiver = ax.quiver(xs,ys,vxs,vys,
                               angles="xy",scale_units="xy",
                               scale=1.5,color="blue",alpha=0.5)
        # print(f"blue_quiver: {blue_quiver}")
        elems.append(blue_quiver)


# MCLによる大域的自己位置推定
class GlobalMcl(Mcl):
    def __init__(self,
                 envmap,
                 num,
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14,
                 direction_dev=0.05): # ロボットの初期姿勢を与えない
        
        super().__init__(envmap, np.array([0,0,0]).T, num, motion_noise_stds,
                         distance_dev_rate, direction_dev)
        # ランダムに姿勢を初期化(なるべく一様分布)
        for p in self.particles:
            p.pose = np.array([
                np.random.uniform(-5.0,5.0),
                np.random.uniform(-5.0,5.0),
                np.random.uniform(-math.pi,math.pi)]).T
        
    
# リセット付き状態分布用MCL
class ResetMcl(Mcl):
    def __init__(self,
                 envmap,
                 init_pose,
                 num,
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14,
                 direction_dev=0.05):
        super().__init__(envmap, init_pose, num, motion_noise_stds,
                         distance_dev_rate, direction_dev)
        
        # 観測値の周辺尤度(観測数毎)
        self.alphas = {}

    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        
        # alpha値の記録
        alpha = sum([p.weight for p in self.particles])
        obsnum = len(observation)
        if not obsnum in self.alphas:
            self.alphas[obsnum] = []
        self.alphas[obsnum].append(alpha)

        self.set_ml() # 最尤パーティクルをセット
        self.resampling() # ここで重みの合計は1になる


    