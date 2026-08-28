
import os
import sys
import math
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pprint import pprint


from goal import Goal
from policy import PolicyEvaluator, DynamicProgramming
from obstacle import Puddle

def check():
    class PolicyEvaluatorTmp:
        def __init__(self, widths, lowerleft=np.array([-4,-4]).T, upperright=np.array([4,4]).T):
            self.pose_min = np.r_[lowerleft, 0] # 連結(min_x, min_y, min_theta)
            self.pose_max = np.r_[upperright, math.pi*2] # theta: 0 ~ 2pi
            self.widths = widths

            self.index_nums = ((self.pose_max - self.pose_min) / self.widths).astype(int)

    pe = PolicyEvaluatorTmp(widths=np.array([0.2, 0.2, math.pi/18]).T)
    print(pe.index_nums) # 区間の個数を表示

    # 様々な座標のインデックス
    pose = np.array([-4,-4,0]).T
    print(np.floor((pose - pe.pose_min)/pe.widths).astype(int))
    pose = np.array([2.9,-2,math.pi]).T
    print(np.floor((pose - pe.pose_min)/pe.widths).astype(int))
    pose = np.array([-5,-2,math.pi/6]).T
    print(np.floor((pose - pe.pose_min)/pe.widths).astype(int))


def value_eval():
    pe = PolicyEvaluator(widths=np.array([0.2,0.2,math.pi/18]).T, goal=Goal(-3,-3))
    v = pe.value_function[:,:,0]
    sns.heatmap(np.rot90(v), square=False) # x軸が行,y軸が列になっているので、左に90回転させて、奥行きがy軸、横がx軸に変更(世界座標系に一致)
    plt.show()

def flag_eval():
    pe = PolicyEvaluator(widths=np.array([0.2,0.2,math.pi/18]).T, goal=Goal(-3,-3))
    f = pe.final_state_flags[:,:,0]
    sns.heatmap(np.rot90(f), square=False)
    plt.show()

def value_by_policy():
    # 方策(制御指示：速度、角速度)
    pe = PolicyEvaluator(np.array([0.2, 0.2, math.pi/18]).T, Goal(-3,-3)) 
    p = np.zeros(pe.index_nums) # 離散状態(px,py,theta)における方策値
    for i in pe.indexes:
        p[i] = sum(pe.policy[i]) # (px,py,theta,u[速度,角速度]) [L, N, M, 2] -> [L, N, M, 1]
        pass # 直進時の状態価値V(s)=0.2, 左回転時の状態価値V(s)=0.5, 右回転時の状態価値V(s)=-0.5

    # thetaは10deg単位で増減
    sns.heatmap(np.rot90(p[:,:,18]), square=False) # 180deg ~ 190degの向きの時の行動を表示
    plt.show()

def gen_state_transition_probs():
    # 状態遷移確率 P(s'|s,a) の確率分布を作成
    vx_unit, vy_unit, th_unit = 0.2, 0.2, math.pi/18
    widths = np.array([vx_unit, vy_unit, th_unit]).T
    goal = Goal(-3,-3)
    time_interval = 0.1
    sampling_num = 10
    pe = PolicyEvaluator(widths, goal, [], time_interval, sampling_num)
    pprint(pe.state_transition_probs)

def reweard_map_by_puddles():
    # 水たまりの深さから報酬を作成(マイナス報酬)
    puddles = [
        Puddle((-2,0),(0,2),depth=0.1),
        Puddle((-0.5,-2),(2.5,1),depth=0.1)
    ]

    vx_unit, vy_unit, th_unit = 0.2, 0.2, math.pi/18
    widths = np.array([vx_unit, vy_unit, th_unit]).T
    goal = Goal(-3,-3)
    time_interval = 0.1
    sampling_num = 10

    pe = PolicyEvaluator(widths, goal, puddles, time_interval, sampling_num)

    sns.heatmap(np.rot90(pe.depths), square=False)
    plt.show()

def value_map_by_sweep():
    # ロボットの行動(右上から左下へ水たまりを無視して突っ切る)に対する状態価値関数Vπを計算

    # 水たまり
    puddles = [
        Puddle(lowerleft=(-2,0), upperright=(0,2), depth=0.1),
        Puddle(lowerleft=(-0.5,-2), upperright=(2.5,1), depth=0.1),
    ]

    # 方策評価
    pe = PolicyEvaluator(widths=np.array([0.2, 0.2, math.pi/18]), # [m,m,rad]
                         goal=Goal(-3,-3),
                         puddles=puddles, # 環境に対する報酬設定
                         time_interval=0.1, # [sec]
                         sampling_num=10,
                         puddle_coef=100,
                         lowerleft=np.array([-4,-4]).T, # 環境の状態空間の終端
                         upperright=np.array([4,4]).T, # 環境の状態空間の終端
    )

    # スイープ(状態価値の計算)の回数
    counter = 0

    for i in range(1, 51, 1):
        pe.policy_evaluation_sweep()
        counter += 1

        if i == 10 or i == 20 or i == 30 or i == 40 or i == 50:
            # 状態価値マップ
            v = pe.value_function[:, :, 18]
            ax = sns.heatmap(np.rot90(v), square=False)
            ax.set_title(f"sweep-count: {counter}", fontsize=14, fontweight="bold", pad=12)
            plt.show()


def value_evaluation_by_repeat_policy_evaluation():
    """反復方策評価による状態価値関数Vπ(s)の計算"""
    # ロボットの行動(右上から左下へ水たまりを無視して突っ切る)に対する状態価値関数Vπを計算
    
    # 水たまり
    puddles = [
        Puddle(lowerleft=(-2,0), upperright=(0,2), depth=0.1),
        Puddle(lowerleft=(-0.5,-2), upperright=(2.5,1), depth=0.1),
    ]

    # 方策評価
    pe = PolicyEvaluator(widths=np.array([0.2, 0.2, math.pi/18]), # [m,m,rad]
                            goal=Goal(-3,-3),
                            puddles=puddles, # 環境に対する報酬設定
                            time_interval=0.1, # [sec]
                            sampling_num=10,
                            puddle_coef=100,
                            lowerleft=np.array([-4,-4]).T, # 環境の状態空間の終端
                            upperright=np.array([4,4]).T, # 環境の状態空間の終端
    )
    
    delta = 1e100
    counter = 0

    # 方策評価を反復実行
    while delta > 0.01:
        delta = pe.policy_evaluation_sweep() # 方策評価
        counter += 1
        print(f"sweep-count: {counter}, max-delta-V: {delta:.3f}")

    # 状態価値マップ
    v_map = pe.value_function[:, :, 18]
    ax = sns.heatmap(np.rot90(v_map), square=False)
    ax.set_title(f"sweep-count: {counter}, delta: {delta:.3f}", fontsize=14, fontweight="bold", pad=12)
    plt.show()

    # Note: 方策評価を終えたタイミングでは、最適方策ではない。

    # 方策の保存
    with open(os.path.sep.join([os.getcwd(), "puddle_ignore_policy_of_policy_evaluation.txt"]), "w") as f:
        for index in pe.indexes:
            p = pe.policy[index]
            f.write("sx:{} sy:{} st:{} uv:{} uw:{}\n".format(index[0],index[1],index[2],p[0],p[1]))

    # 状態価値の保存
    with open(os.path.sep.join([os.getcwd(), "puddle_ignore_values_of_policy_evaluation.txt"]), "w") as f:
        for index in pe.indexes:
            v = pe.value_function[index]
            f.write("sx:{} sy:{} st:{} V:{}\n".format(index[0], index[1], index[2], v))

    print("Fin Policy-Evaluation")


def value_evaluation_by_value_iteration():
    """価値反復法による状態価値関数Vπ(s)の計算"""
    # ロボットの行動(右上から左下へ水たまりを無視して突っ切る)に対する状態価値関数Vπを計算
        
    # 水たまり
    puddles = [
        Puddle(lowerleft=(-2,0), upperright=(0,2), depth=0.1),
        Puddle(lowerleft=(-0.5,-2), upperright=(2.5,1), depth=0.1),
    ]

    # 動的計画法による価値反復
    dp = DynamicProgramming(widths=np.array([0.2, 0.2, math.pi/18]), # [m,m,rad]
                            goal=Goal(-3,-3),
                            puddles=puddles, # 環境に対する報酬設定
                            time_interval=0.1, # [sec]
                            sampling_num=10,
                            puddle_coef=100,
                            lowerleft=np.array([-4,-4]).T, # 環境の状態空間の終端
                            upperright=np.array([4,4]).T, # 環境の状態空間の終端
    )

    delta = 1e100
    counter = 0 # スイープの回数

    while delta > 0.104:
        delta = dp.value_iteration_sweep()
        counter += 1
        print(f"sweep-count: {counter}, max-delta-V: {delta}")

    # 状態価値マップ
    v_map = dp.value_function[:, :, 18]
    ax_v = sns.heatmap(np.rot90(v_map), square=False)
    ax_v.set_title(f"[VMap] sweep-count: {counter}, delta: {delta:.3f}", fontsize=14, fontweight="bold", pad=12)
    plt.show()

    # Note: 価値反復を終えたタイミングでは、最適方策を得ることができる。

    # 方策の保存
    with open(os.path.sep.join([os.getcwd(), "puddle_ignore_policy_of_value_iteration.txt"]), "w") as f:
        for index in dp.indexes:
            p = dp.policy[index]
            f.write("sx:{} sy:{} st:{} uv:{} uw:{}\n".format(index[0],index[1],index[2],p[0],p[1]))

    with open(os.path.sep.join([os.getcwd(), "policy.txt"]), "w") as f:
            for index in dp.indexes:
                p = dp.policy[index]
                f.write("{} {} {} {} {}\n".format(index[0],index[1],index[2],p[0],p[1]))
        
    # 状態価値の保存
    with open(os.path.sep.join([os.getcwd(), "puddle_ignore_values_of_value_iteration.txt"]), "w") as f:
        for index in dp.indexes:
            v = dp.value_function[index]
            f.write("sx:{} sy:{} st:{} V:{}\n".format(index[0], index[1], index[2], v))

    with open(os.path.sep.join([os.getcwd(), "value.txt"]), "w") as f:
            for index in dp.indexes:
                v = dp.value_function[index]
                f.write("{} {} {} {}\n".format(index[0], index[1], index[2], v))

    # 方策マップ
    p = np.zeros(dp.index_nums)
    for i in dp.indexes:
        p[i] = sum(dp.policy[i]) # 指令速度と指令角速度を足すと 1.0: 直進, 2.0: 左回転, -2.0: 右回転

    p_map = p[:, :, 18]
    ax_p = sns.heatmap(np.rot90(p_map), square=False) # 180〜190[deg]の向きのときの行動を図示
    ax_p.set_title(f"[PMap] sweep-count: {counter}, delta: {delta:.3f}", fontsize=14, fontweight="bold", pad=12)
    plt.show()

    print("Fin Value-Iteration")


if __name__ == "__main__":
    # check()
    # value_eval()
    # flag_eval()
    # value_by_policy()
    # gen_state_transition_probs()
    # reweard_map_by_puddles()
    # value_map_by_sweep()
    # value_evaluation_by_repeat_policy_evaluation()
    value_evaluation_by_value_iteration()