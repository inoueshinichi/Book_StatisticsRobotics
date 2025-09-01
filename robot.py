
import math

import numpy as np
from scipy.stats import expon, norm, uniform
import matplotlib.pyplot as plt

import matplotlib.patches as patches
import matplotlib.animation as anm


class IdealRobot:
    def __init__(self, 
                 pose,          # ロボットの姿勢 (x, y, Θ)
                 agent=None,    # ロボットのエージェント
                 sensor=None,
                 color='black', # ロボットの描画色
                 ):
        
        self.pose = pose
        self.r = 0.2         # 描画用のロボット半径
        self.color = color
        self.agent = agent
        self.poses = [pose]  # ロボットの軌跡
        self.sensor = sensor # センサー

    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta) # ロボット鼻先のX座標
        yn = y + self.r * math.sin(theta) # ロボット鼻先のY座標

        elems += ax.plot([x,xn], [y,yn], color=self.color) # 鼻先ベクトル
        c = patches.Circle(xy=(x,y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))

        # ロボットの移動軌跡の描画
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")

        # ロボットからランドマークまでの距離と方角を描画
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])

        # Agentの描画
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)

    # 状態遷移関数(状態方程式)
    @classmethod
    def state_transition(cls,
                         nu,    # 速度 v_t
                         omega, # 角速度 ω_t
                         time,  # Δt
                         pose,  # 時刻tでの(x_t-1, y_t-1, θ_t-1)
                         ):
        """
        状態方程式
        入力:
            + 現在時刻t-1の状態: (x_t-1, y_t-1, θ_t-1)
            + Δtにおける変化量: (v_t, ω_t)

        出力
            + 次の時刻tの状態: (x_t, y_t, θ_t)
        """
        t0 = pose[2] # θ_t-1
        if math.fabs(omega) < 1e-10: # 角速度がほぼゼロの場合
            return pose + np.array([
                    nu * math.cos(t0), # v_t(x) * cos(θ_t-1) = Δx
                    nu * math.sin(t0), # v_t(y) * sin(θ_t-1) = Δy 
                    omega,             # ω_t = Δθ
                ])
        
        else:
            return pose + np.array([
                nu / omega * (math.sin(t0 + omega*time) - math.sin(t0)),      # Δx = v_t(x) / ω_t * (sin(θ_t-1 + ω_t * Δt) - sin(θ_t-1))
                nu / omega * (-1 * math.cos(t0 + omega*time) + math.cos(t0)), # Δy = v_t(y) / ω_t * (-cos(θ_t-1 + ω_t * Δt) + cos(θ_t-1))
                omega * time                                                  # Δθ = ω_t * Δt
            ])
        
    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        if self.sensor: self.sensor.data(self.pose)

        

class Robot(IdealRobot):
    def __init__(self, 
                 pose, 
                 agent=None, 
                 sensor=None, 
                 color='black',
                 noise_per_meter=5,           # 1[m]あたりの小石の数
                 noise_std=math.pi/60,        # 小石を踏んだ時にロボットの向きΘ[deg]に発生する雑音の標準偏差 
                 bias_rate_stds=(0.1,0.1),    # (nu,omega)に対するバイアス誤差
                 expected_stuck_time=1e-100,  # スタックするまでの平均時間
                 expected_escape_time=1e-100, # スタックから抜け出すまでの平均時間
                 expected_kidnap_time=1e-100, # 誘拐が発生するまでの平均時間
                 kidnap_range_x=(-5.0,5.0),   # ワープ範囲(x)
                 kidnap_range_y=(-5.0,5.0),   # ワープ範囲(y)
                 ):
        
        super().__init__(pose, agent, sensor, color)

        # ノイズ
        if noise_per_meter and noise_std:
            self.is_state_noise = True
            self.noise_pdf = expon(scale=1.0/(1e-100 + noise_per_meter)) # 小石を踏むまでの平均道のり. scaleは1/λ(平均道のり)
            self.distance_until_noise = self.noise_pdf.rvs()
            self.theta_noise = norm(scale=noise_std)
        else:
            self.is_state_noise = False
            self.noise_pdf = None
            self.distance_until_noise = None
            self.theta_noise = None

        # 制御指令バイアス
        if bias_rate_stds:
            self.is_control_bias = True
            self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0])    # 速度バイアス
            self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1]) # 角速度バイアス
        else:
            self.is_control_bias = False
            self.bias_rate_nu = None
            self.bias_rate_omega = None

        # ロボットのスタック
        if expected_stuck_time and expected_escape_time:
            self.is_robot_stuck = True
            self.stuck_pdf = expon(scale=expected_stuck_time)
            self.escape_pdf = expon(scale=expected_escape_time)
            self.time_until_stuck = self.stuck_pdf.rvs()
            self.time_until_escape = self.escape_pdf.rvs()
            self.is_stuck = False
        else:
            self.is_robot_stuck = False
            self.stuck_pdf = None
            self.escape_pdf = None
            self.time_until_stuck = None
            self.time_until_escape = None
            self.is_stuck = None

        # 誘拐
        if expected_kidnap_time and kidnap_range_x and kidnap_range_y:
            self.is_robot_kidnap = True
            self.kidnap_pdf = expon(scale=expected_kidnap_time)
            self.time_until_kidnap = self.kidnap_pdf.rvs()
            rx, ry = kidnap_range_x, kidnap_range_y
            self.kidnap_dist = uniform(loc=(rx[0],ry[0],0.0), scale=(rx[1]-rx[0], ry[1]-ry[0], 2*math.pi))
        else:
            self.is_robot_kidnap = False
            self.kidnap_pdf = None
            self.time_until_kidnap = None
            rx, ry = None, None
            self.kidnap_dist = None


    def noise(self, pose, nu, omega, time_interval):
        self.distance_until_noise -= abs(nu)*time_interval + self.r*abs(omega)*time_interval # -1*(直進成分 + 回転成分)
        if self.distance_until_noise <= 0.0:
            self.distance_until_noise += self.noise_pdf.rvs() # 端数を残すために =ではなく+=
            pose[2] += self.theta_noise.rvs()

        return pose
    
    def bias(self, nu, omega):
        return nu*self.bias_rate_nu, omega*self.bias_rate_omega
    
    def stuck(self, nu, omega, time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True

        return nu*(not self.is_stuck), omega*(not self.is_stuck)
    
    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose

    # override 
    def one_step(self, time_interval):

        if not self.agent: 
            return
        
        # 観測
        obs = self.sensor.data(self.pose) if self.sensor else None

        # 制御指令
        nu, omega = self.agent.decision(obs) 

        # 制御値に対するバイアス
        if self.is_control_bias:
            nu, omega = self.bias(nu, omega) 

        # スタック
        if self.is_robot_stuck:
            nu, omega = self.stuck(nu, omega, time_interval) 

        # 更新(理想)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)

        # ノイズ
        if self.is_state_noise:
            self.pose = self.noise(self.pose, nu, omega, time_interval) 

        # 誘拐
        if self.is_robot_kidnap:
            self.pose = self.kidnap(self.pose, time_interval) 
            if self.sensor: self.sensor.data(self.pose)



