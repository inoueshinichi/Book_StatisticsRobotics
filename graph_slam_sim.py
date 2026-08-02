"""
グラフSlamによるシミュレーション
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import os

from world import World
from map import Map
from landmarks import Landmark
from robot import Robot
from sensor import PsiCamera
from agent import LoggerAgent




def logging_pose_of_landmark():
    # 観測毎にランドマークの姿勢(センサ値)
    # (time, phi, x, y, theta)を取得
    # phi = カメラとランドマークの相対角度
    time_interval = 3
    world = World(180, time_interval, debug=False)

    # 真の地図
    m = Map()
    landmark_positions =  [(-4,2), (2,-3), (3,3), (0,4), (1,1), (-3,-1)]
    for p in landmark_positions:
        m.append_landmark(Landmark(*p))
    world.append(m)

    # ロボット
    init_pose = np.array([0.0, -3.0, 0.0]).T
    agent = LoggerAgent(
        nu=0.2, # 直線速度
        omega=5.0/180*math.pi, # 角速度
        interval_time=time_interval, # 更新間隔
        init_pose=init_pose, # 初期値
    )
    robot = Robot(init_pose, sensor=PsiCamera(m), agent=agent, color="red")
    world.append(robot)

    world.draw()


def make_graph_slam_from_logger():
    # graph_slam_log_1.txtのデータ
    # (time, phi, x, y, theta)
    # からグラフスラムを作成

    def make_ax():
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        return ax
    
    def draw_trajectory(xs, ax):
        # 軌跡の描画
        poses = [xs[s] for s in range(len(xs))]
        ax.scatter([e[0] for e in poses], 
                   [e[1] for e in poses],
                   s=5, marker=".", color="black")
        ax.plot([e[0] for e in poses], 
                [e[1] for e in poses],
                linewidth=0.5, color='black')
        
    def draw_observations(xs, zlist, ax):
        # センサ値の描画
        for s in range(len(xs)):
            if s not in zlist:
                continue
        
            for obs in zlist[s]:
                x,y,theta = xs[s]
                ell = obs[1][0] # センサ値(極座標でのランドマークまでの距離)
                phi = obs[1][1] # センサ値(極座標でのランドマークの角度)
                mx = x + ell * math.cos(theta + phi)
                my = y + ell * math.sin(theta + phi)
                ax.plot([x,mx], [y,my], color='pink', alpha=0.5)
        
    def draw(xs, zlist):
        ax = make_ax()
        draw_observations(xs, zlist, ax)
        draw_trajectory(xs, ax)
        plt.show()

    def read_data():
        # graph_slam_log_1.txtから読み込み
        hat_xs = {} # 軌跡データ
        zlist = {} # センサ値

        current_dir = Path(__file__).resolve().parent
        file_path = os.path.join(str(current_dir), "graph_slam_log_1.txt")
        with open(file_path, "r") as f:
            for line in f.readlines():
                tmp = line.rstrip().split()
                # x step rx, ry, rtheta : ロボット
                # z step phi, sx, sy, stheta : ランドマーク

                step = int(tmp[1])
                if tmp[0] == "x": #ロボットの姿勢
                    hat_xs[step] = np.array([
                        float(tmp[2]), # rx
                        float(tmp[3]), # ry
                        float(tmp[4]), # rtheta
                    ]).T
                elif tmp[0] == "z": #センサ値
                    if step not in zlist: # まだ辞書が空の時はからの辞書を作る
                        zlist[step] = []
                    zlist[step].append((
                        int(tmp[2]), # ランドマークID
                        np.array([
                            float(tmp[3]), # phi(カメラとランドマークの相対角度)
                            # 下記、極座標系のセンサ値
                            float(tmp[4]), # ell(カメラ座標におけるランドマークまでの距離)
                            float(tmp[5]), # theta(カメラ座標系におけるランドマークの角度)
                        ])
                    ))

            # ロボットの軌跡, センサ値
            return hat_xs, zlist

    # ロギングデータ読み込み
    hat_xs, zlist = read_data()

    from pprint import pprint
    pprint(f"hat_xs:\n {hat_xs}")
    pprint(f"zlist:\n {zlist}")

    # 描画
    draw(hat_xs, zlist)


if __name__ == "__main__":
    # logging_pose_of_landmark()
    make_graph_slam_from_logger()