# 3.3 ロボットの観測

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as anm


class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x,y]).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))


class Map:
    def __init__(self):
        self.landmarks = []

    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    def draw(self, ax, elems):
        for lm in self.landmarks:
            lm.draw(ax, elems)


class World:
    def __init__(self, 
                 time_span: float,     # シミュレーションタイムduration
                 time_interval: float, # 1コマの時間Δt
                 debug=False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval

    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(4,4)) # 4x4 inch図
        ax = fig.add_subplot(111)
        ax.set_aspect('equal') # 縦横比を座標の値と一致させる
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)

        elems = []

        if self.debug:
            for i in range(1000): 
                self.one_step(i, elems, ax) # デバッグ時はアニメーションさせない
        else:
            print('Start animation')
            
            # シミュレーション時間とタイムスパンからアニメーションを作成
            self.ani = anm.FuncAnimation(fig,
                                         self.one_step,
                                         fargs=(elems, ax),
                                         frames=int(self.time_span /self.time_interval ) + 1,
                                         interval=int(self.time_interval * 1000),
                                         repeat=False)
            
            print('objects', self.objects)
            print('elems ', elems)

            plt.show()


    def one_step(self, 
                 i,     # ステップ番号 
                 elems, # 描画リスト
                 ax,    # サブプロット
                 ):
        # アニメーションを1コマ進めるメソッド
        while elems: 
            elems.pop().remove() # 二重描画を防止
        time_str = "t = %.2f[s]" % (self.time_interval * i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega
    

class IdealRobot:
    def __init__(self, 
                 pose,          # ロボットの姿勢 (x, y, Θ)
                 agent=None,    # ロボットのエージェント
                 color='black', # ロボットの描画色
                 ):
        
        self.pose = pose
        self.r = 0.2        # 描画用のロボット半径
        self.color = color
        self.agent = agent
        self.poses = [pose] # ロボットの軌跡

    def one_step(self, time_interval):
        if not self.agent: return
        nu, omega = self.agent.decision()
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)

    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta) # ロボット鼻先のX座標
        yn = y + self.r * math.sin(theta) # ロボット鼻先のY座標

        elems += ax.plot([x,xn], [y,yn], color=self.color) # 鼻先ベクトル
        c = patches.Circle(xy=(x,y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))

        # 軌跡の描画
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")

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
                nu * math.sin(t0), # v_t(x) * sin(θ_t-1) = Δx
                nu * math.cos(t0), # v_t(y) * cos(θ_t-1) = Δy
                omega,             # ω_t = Δθ
            ])
        
        else:
            return pose + np.array([
                nu / omega * (math.sin(t0 + omega*time) - math.sin(t0)),      # Δx = v_t(x) / ω_t * (sin(θ_t-1 + ω_t * Δt) - sin(θ_t-1))
                nu / omega * (-1 * math.cos(t0 + omega*time) + math.cos(t0)), # Δy = v_t(y) / ω_t * (-cos(θ_t-1 + ω_t * Δt) + cos(θ_t-1))
                omega * time                                                  # Δθ = ω_t * Δt
            ])
        

class IdealCamera:
    def __init__(self, env_map):
        self.map = env_map

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            p = self.observation_function(cam_pose, lm.pos)
            observed.append((p, lm.id))

        return observed
    
    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi
        return np.array([np.hypot(*diff), phi]).T


def case1():
    world = World(10, 0.1)

    m = Map()
    m.append_landmark(Landmark(2,-2))  # id: 0
    m.append_landmark(Landmark(-1,-3)) # id: 1
    m.append_landmark(Landmark(3,3))   # id: 2
    world.append(m)

    agent = Agent(0.0, 0.0) # 0.2 [m/s] で直進
    robot = IdealRobot(pose=np.array([0, 0, 0]).T, agent=None, color="blue")
    world.append(robot)

    world.draw()


if __name__ == "__main__":
    case1()