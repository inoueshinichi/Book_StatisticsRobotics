"""
Monte Carlo Localization(MCL)によるパーティクルフィルタ
"""
import math
import numpy as np
from scipy.stats import multivariate_normal

from robot import IdealRobot

class Particle:
    def __init__(self, init_pose):
        self.pose = init_pose

    def motion_update(self,
                      nu,
                      omega,
                      time,
                      noise_rate_pdf, # 各ステップごとに確率分布が更新される
                      ):
        ns = noise_rate_pdf.rvs() # サンプリング (nn,no,on,oo)
        noised_nu = nu + ns[0] * math.sqrt(abs(nu)/time) + ns[1] * math.sqrt(abs(omega)/time) # 移動量 = 前ステップの移動量 * ノイズ割合
        noised_omega = omega + ns[2] * math.sqrt(abs(nu)/time) + ns[3] * math.sqrt(abs(omega)/time) # 回転量 = 前ステップの回転量 * ノイズ割合
        
        # 状態遷移方程式で現在の状態に更新
        self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)


class Mcl:
    def __init__(self, 
                 init_pose, 
                 num,
                 motion_noise_stds):
        self.particles = [Particle(init_pose) for i in range(num)]

        v = motion_noise_stds
        c = np.diag([v["nn"]**2,v["no"]**2,v["on"]**2,v["oo"]**2]) # 速度と角速度の誤差の分散共分散行列
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)    # 多次元ガウス分布

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

