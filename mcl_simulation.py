"""
Monte Carlo Localization (MCL)を用いた
パーティクルフィルタによる自己位置推定
"""
import copy
import math
import inspect
from pprint import pprint
import numpy as np
import pandas as pd

from world import World
from map import Map
from landmarks import Landmark
from robot import Robot, IdealRobot
from sensor import IdealCamera, Camera
from agent import CommandAgent, EstimationAgent
from estimator import (
    MclParticleFilterEstimator,
    GlobalMclParticleFilterEstimator,
    ResetMclParticleFilterEstimator,
    KldMclParticleFilterEstimator,
)
from mcl import (
    Mcl,
    GlobalMcl,
    ResetMcl,
)


def mcl1_pattern():
    """単純なMCLパーティクルフィルタによる自己位置推定"""
    world = World(60, 0.1)

    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    init_pose = np.array([2,2,math.pi/6]).T
    circling = EstimationAgent(time_interval=None, 
                               nu=0.2, 
                               omega=10.0/180*math.pi, 
                               estimator=None)
    r = Robot(init_pose, 
              sensor=Camera(m), 
              agent=circling, 
              expected_kidnap_time=None)
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def mcl2_pattern():
    world = World(times_span=30, time_interval=0.1)

    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    init_pose = np.array([2,2,math.pi/6]).T
    mcl_pf_e = MclParticleFilterEstimator(envmap=m, 
                                          init_pose=init_pose, 
                                          num=100)
    
    circling = EstimationAgent(time_interval=None,
                               nu=0.4, 
                               omega=20.0/180*math.pi, 
                               estimator=mcl_pf_e) 
    
    r = Robot(init_pose, 
              sensor=Camera(m), 
              agent=circling, 
              expected_kidnap_time=None)
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def mcl3_pattern():
    init_pose = np.array([0,0,0]).T
    
    native_estimator = Mcl(envmap=None, 
                           init_pose=init_pose,
                           num=100, 
                           motion_noise_stds={
                               "nn":0.01, 
                               "no":0.02, 
                               "on":0.03, 
                               "oo":0.04
                            })
    
    a = EstimationAgent(time_interval=0.1, 
                        nu=0.2,
                        omega=10.0/180*math.pi,
                        estimator=native_estimator)

    native_estimator.motion_update(0.2, 10.0/180*math.pi, time=0.1)

    for p in native_estimator.particles:
        print(p.pose)


def mcl4_pattern():
    time_interval = 0.1 # 0.1[s]
    world = World(60, time_interval)

    init_pose = np.array([0,0,0]).T

    estimator = MclParticleFilterEstimator(envmap=None, 
                                           init_pose=init_pose,
                                           num=100,
                                           motion_noise_stds={
                                               "nn":0.01, 
                                               "no":0.02, 
                                               "on":0.03, 
                                               "oo":0.04
                                           })


    circling = EstimationAgent(time_interval=time_interval,
                               nu=0.2,
                               omega=10.0/180*math.pi,
                               estimator=estimator)
    
    r = Robot(init_pose, sensor=None, agent=circling, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def mcl5_pattern(motion_noise_stds):
    time_interval = 0.1 # 0.1[s]
    world = World(time_span=60, time_interval=time_interval)

    init_pose = np.array([0,0,0]).T
    estimator = MclParticleFilterEstimator(envmap=None, 
                                           init_pose=init_pose, 
                                           num=100, 
                                           motion_noise_stds=motion_noise_stds)
    
    circling = EstimationAgent(time_interval=time_interval,
                               nu=0.2,
                               omega=10.0/180*math.pi,
                               estimator=estimator)
    
    r = Robot(init_pose, sensor=None, agent=circling, 
              expected_kidnap_time=None, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)




def motion_test_forward():
    # ロボットの動きをブラックボックスと見立てて,
    # 繰り返し実験からロボットの雑音パラメータを決定する作業

    world = World(time_span=30, time_interval=0.1)

    initial_pose = np.array([0,0,0]).T
    robots = []
    r = Robot(initial_pose, sensor=None, 
              agent=CommandAgent(nu=0.1, omega=0.0))

    for _ in range(100):
        copy_r = copy.copy(r)
        copy_r.distance_until_noise = copy_r.noise_pdf.rvs() # 最初に雑音が発生するタイミングを変える
        world.append(copy_r)
        robots.append(copy_r)   

    world.draw(title=inspect.currentframe().f_code.co_name) 


def motion_test_forward_bias():
    world = World(time_span=30.0, time_interval=0.1)
    initial_pose = np.array([0,0,0]).T
    robots = []

    for _ in range(100):
        r = Robot(initial_pose, sensor=None, 
                  agent=CommandAgent(nu=0.1, omega=0.0)) # ここで生成されるロボットは異なるバイアスを持つ
        world.append(r)
        robots.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)

    poses = pd.DataFrame([
        [math.sqrt(r.pose[0]**2 + r.pose[1]**2), r.pose[2]] for r in robots], 
        columns=['r', 'theta'])
    print(poses.transpose())

    print(poses["r"].var()) 
    print(poses["r"].mean())
    print(math.sqrt(poses["r"].var()/poses["r"].mean()))


def motion_test_rot_bias():
    world = World(time_span=30.0, time_interval=0.1)

    init_pose = np.array([0,0,0]).T
    robots = []

    for _ in range(100):
        r = Robot(init_pose, sensor=None, agent=CommandAgent(nu=0, omega=0.1))
        world.append(r)
        robots.append(r)
    
    world.draw(title=inspect.currentframe().f_code.co_name)

    poses = pd.DataFrame([[math.sqrt(r.pose[0]**2 + r.pose[1]**2), r.pose[2]] 
                          for r in robots], columns=['r', 'theta'])
    poses.transpose()

    print(poses["theta"].var())
    print(poses["theta"].mean())
    math.sqrt(poses["theta"].var()/poses["theta"].mean())


def mcl6_pattern():
    time_interval = 0.1 # 0.1[s]
    world = World(time_span=40.0, time_interval=time_interval)

    init_pose = np.array([0,0,0]).T

    e = MclParticleFilterEstimator(envmap=None,
                                    init_pose=init_pose,
                                    num=100,
                                    motion_noise_stds={
                                        "nn":0.001, 
                                        "no":0.001, 
                                        "on":0.13, 
                                        "oo":0.001
                                    })
    
    a = EstimationAgent(time_interval=time_interval,
                        nu=0.1,
                        omega=0.0,
                        estimator=e)

    r = Robot(pose=init_pose, sensor=None, agent=a, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def mcl7_pattern():
    time_interval = 0.1
    world = World(40.0, time_interval)

    init_pose = np.array([0,0,0]).T

    estimator = MclParticleFilterEstimator(envmap=None,
                                           init_pose=init_pose,
                                           num=100,
                                           motion_noise_stds={
                                               "nn":0.001, 
                                               "no":0.001, 
                                               "on":0.13, 
                                               "oo":0.001
                                           })
    
    circling = EstimationAgent(time_interval, 0.2, 10.0/180*math.pi, estimator)
    r = Robot(init_pose, sensor=None, agent=circling, color="red" )
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def mcl8_pattern():
    time_interval = 0.1

    world = World(time_span=40.0, time_interval=time_interval)
    init_pose = np.array([0,0,0]).T

    for _ in range(100):
        r = Robot(pose=init_pose, 
                  sensor=None, 
                  agent=CommandAgent(nu=0.2, omega=10.0/180*math.pi), 
                  color="grey")
        world.append(r)
    
    world.draw(title=inspect.currentframe().f_code.co_name)


def mcl9_pattern():
    time_interval = 0.1
    world = World(time_span=40.0, time_interval=time_interval)

    # 地図を生成して3つのランドマークを追加
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボットを作る
    init_pose = np.array([0,0,0]).T
    estimator = MclParticleFilterEstimator(envmap=m,
                                           init_pose=init_pose,
                                           num=100)


    a = EstimationAgent(time_interval=time_interval, 
                        nu=0.2, 
                        omega=10.0/180*math.pi, 
                        estimator=estimator)
    
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def mcl10_pattern():
    time_interval = 0.5
    world = World(time_span=40.0, time_interval=time_interval)

    # 地図を生成して3つのランドマークを追加
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # 初期状態
    init_pose = np.array([0,0,0]).T

    # 推定器
    e = MclParticleFilterEstimator(envmap=m,
                                   init_pose=init_pose,
                                   num=100)
    
    a = EstimationAgent(time_interval=time_interval, 
                        nu=0.2, 
                        omega=10.0/180*math.pi, 
                        estimator=e)
    
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def sensor_experiment():
    m = Map()
    m.append_landmark(Landmark(1,0))

    distance = []
    direction = []
    

    for _ in range(100):
        c = Camera(m) # バイアスの影響も考慮するために毎回カメラを新規作成
        cam_pose = np.array([0,0,0]).T
        d = c.data(cam_pose) # カメラ位置
        if len(d) > 0:
            distance.append(d[0][0][0])
            direction.append(d[0][0][1])
    
    df = pd.DataFrame()
    df["distance"] = distance
    df["direction"] = direction

    pprint(df)
    print()
    print('>>>統計情報>>>')
    print(f"平均(距離,角度): \n{df.mean()}") # 平均(距離,角度)
    print(f"標準偏差(距離,角度): \n{df.std()}") # 標準偏差(距離,角度)
    

def mcl11_pattern():
    time_interval = 0.1
    world = World(time_span=40, time_interval=time_interval, debug=False)

    # 地図 & ランドマーク
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボット
    init_pose = np.array([0,0,0]).T
    e = MclParticleFilterEstimator(envmap=m, init_pose=init_pose, num=100)
    a = EstimationAgent(time_interval=time_interval, 
                        nu=0.2, omega=10.0/180*math.pi, 
                        estimator=e)
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def mcl12_pattern():
    time_interval = 0.1
    world = World(time_span=40, time_interval=time_interval, debug=False)

    # 地図&ランドマーク
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボット
    init_pose = np.array([0,0,0]).T
    e = MclParticleFilterEstimator(envmap=m, init_pose=init_pose, num=200)
    a = EstimationAgent(time_interval, nu=0.5, omega=20.0/180*math.pi, estimator=e)
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)


def mcl13_pattern():
    # 大域的自己位置推定
    # パーティクルの初期位置とロボットの初期位置を一様分布で初期化
    animation = True
    time_interval = 0.1
    world = World(30, time_interval, debug=not animation)

    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # ロボットの初期位置をランダムに設定
    init_pose = np.array([
        np.random.uniform(-4.0,4.0),
        np.random.uniform(-4.0,4.0),
        np.random.uniform(-math.pi, math.pi),
    ]).T

    gmcle = GlobalMclParticleFilterEstimator(envmap=m, num=100)

    a = EstimationAgent(time_interval, nu=0.5, omega=20.0/180*math.pi, 
                        estimator=gmcle)
    r = Robot(init_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)

    # 真の姿勢と推定姿勢を表示
    print(f"GT pose: {r.pose}, Pred pose: {gmcle.pose}")
    

def mcl14_pattern():
    # 誘拐ロボット問題
    animation = True
    time_interval = 0.1
    world = World(30, time_interval, debug=not animation)

    m = Map()
    for ln in [(-4,2),(2,-3),(3,3)]:
        m.append_landmark(Landmark(*ln))
    world.append(m)

    # パーティクルの初期位置とロボットの初期位置を別にする
    # 各パーティクルの初期姿勢はランダム値だがすべて同じ
    init_pose = np.array([
        np.random.uniform(-5.0,5.0),
        np.random.uniform(-5.0,5.0),
        np.random.uniform(-math.pi, math.pi),
    ]).T
    robot_pose = np.array([
        np.random.uniform(-5.0,5.0),
        np.random.uniform(-5.0,5.0),
        np.random.uniform(-math.pi, math.pi),
    ]).T

    mcle = MclParticleFilterEstimator(envmap=m, init_pose=init_pose, num=100)
    a = EstimationAgent(time_interval, nu=1.0, omega=20.0/180*math.pi, estimator=mcle)
    r = Robot(robot_pose, sensor=Camera(m), agent=a, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)

    # 真の姿勢と推定姿勢を表示
    print(f"GT pose: {r.pose}, Pred pose: {mcle.pose}")


def mcl15_pattern():
    # リセット付き状態分布MCL
    time_interval = 0.1
    world = World(300, time_interval)

    m = Map()                                  
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)

    init_pose = np.array([0,0,0]).T
    pf = ResetMclParticleFilterEstimator(envmap=m, init_pose=init_pose, num=100)
    circling = EstimationAgent(time_interval, 
                               nu=0.5, omega=15.0/180*math.pi, 
                               estimator=pf)
    
    r = Robot(init_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)

    def display_alphas(pf):
        for num in pf.alphas: ###mclalpharesult
            print("landmarks:", num, "particles:", len(pf.particles), "min:", min(pf.alphas[num]), "max:", max(pf.alphas[num]))

    display_alphas(pf)


def mcl16_pattern():
    # リセット付き状態分布MCL
    time_interval = 0.1
    world = World(40, time_interval)

    m = Map()                                  
    m.append_landmark(Landmark(-4,2))
    m.append_landmark(Landmark(2,-3))
    m.append_landmark(Landmark(3,3))
    world.append(m)

    init_pose = np.array([0,0,0]).T
    pf = ResetMclParticleFilterEstimator(envmap=m, init_pose=init_pose, num=100)
    circling = EstimationAgent(time_interval, 
                               nu=0.5, 
                               omega=30.0/180*math.pi, 
                               estimator=pf)
    r = Robot(init_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    world.draw(title=inspect.currentframe().f_code.co_name)

    def display_alphas(pf):
        for num in pf.alphas: ###mclalpharesult
            print("landmarks:", num, "particles:", len(pf.particles), "min:", min(pf.alphas[num]), "max:", max(pf.alphas[num]))

    display_alphas(pf)

if __name__ == "__main__":
    # mcl1_pattern()
    # mcl2_pattern()
    # mcl3_pattern()
    # mcl4_pattern() # パーティクルの分布(エージェントの推定分布)が狭すぎて実機の姿勢が乖離した状態
    # mcl5_pattern({"nn":0.01, "no":0.02, "on":0.03, "oo":0.04}) # パーティクルの分布(エージェントの推定分布)が狭すぎて実機の姿勢が乖離した状態
    # mcl5_pattern({"nn":1, "no":2, "on":3, "oo":4}) # 分布が広すぎて推定できていない状態
    # mcl5_pattern({"nn":0.01, "no":0.01, "on":0.01, "oo": 0.3}) # 回転にノイズが大きい
    # mcl5_pattern({"nn":0.1, "no":0.01, "on":0.01, "oo":0.01}) # 移動にノイズが大きい
    # mcl5_pattern({"nn":0.1, "no":0.01, "on":0.01, "oo":0.001}) # 回転にノイズが小さい
    # mcl5_pattern({"nn":0.02, "no":0.01, "on":0.01, "oo":0.1}) # 移動にノイズが小さい
    # motion_test_forward()
    # motion_test_forward_bias()
    # motion_test_rot_bias()
    # mcl6_pattern()
    # mcl7_pattern()
    # mcl8_pattern()
    # mcl9_pattern()
    # mcl10_pattern()
    # sensor_experiment()
    # mcl11_pattern()
    # mcl12_pattern()
    # mcl13_pattern()
    # mcl14_pattern()
    # mcl15_pattern()
    mcl16_pattern()

