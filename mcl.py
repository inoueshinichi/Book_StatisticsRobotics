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

class Mcl:
    def __init__(self, 
                 envmap,
                 init_pose, 
                 num,
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14,
                 direction_dev=0.05):
        
        # パーティクルの生成
        self.particles = [
            Particle(init_pose, 1.0/num) for i in range(num)
            ]
        
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
    def motion_update(self, 
                      nu, 
                      omega, 
                      time, 
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
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        vys = [math.sin(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        elems.append(ax.quiver(xs,ys,vxs,vys,
                               angles="xy",scale_units="xy",
                               scale=1.5,color="blue",alpha=0.5))

