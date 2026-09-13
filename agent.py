import os
from pathlib import Path
import math
import random
import itertools
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Set, Optional, Union, override

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anm

from robot import IdealRobot
from estimator import Estimator
from goal import Goal

# 制御指令コントローラ
class Agent(ABC):
    
    @abstractmethod
    def decision(self, observation: Optional[np.ndarray] = None) -> Tuple[float,float]:
        raise NotImplementedError()

    @abstractmethod
    def draw(self, ax, elems):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def policy(cls, pose: np.ndarray, goal: Goal) -> Tuple[float,float]:
        raise NotImplementedError()


class CommandAgent(Agent):

    def __init__(self,
                 nu: float, # 速度制御指示
                 omega: float, # 角速度制御指示
                 ):
        super()
        self.nu: float = nu
        self.omega: float = omega

    def decision(self, observation: Optional[np.ndarray] = None) -> Tuple[float,float]:
        return self.nu, self.omega

    def draw(self, ax, elems):
        pass

    def policy(cls, pose: np.ndarray, goal: Goal) -> Tuple[float,float]:
        pass


    
class EstimationAgent(CommandAgent):
    def __init__(self, 
                 time_interval: Optional[float], # Δタイム
                 nu: float, # 速度制御指示 
                 omega: float, # 角速度制御指示
                 estimator: Optional[Estimator],
                 ):
        super().__init__(nu, omega)
        self.estimator: Optional[Estimator] = estimator # 推定器(KF,MCL,KldMCLなど)
        self.time_interval: Optional[float] = time_interval

        # 1ステップ前の状態変数
        self.prev_nu: float = 0.0
        self.prev_omega: float = 0.0

    def decision(self, observation: Optional[np.ndarray] = None) -> Tuple[float,float]:
        """ロボットの自己位置を
        [1] 状態方程式で更新
        [2] 観測方程式で補正
        """
        if self.time_interval is None or self.estimator is None:
            return self.nu, self.omega

        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        return self.nu, self.omega

    def draw(self, ax, elems):
        if self.estimator is None: return

        # print(f"agent draw")
        self.estimator.draw(ax, elems)

        # Write ml
        x, y, t = self.estimator.pose
        s = "({:.2f}, {:.2f}, {})".format(x, y, int(t*180/math.pi)%360)
        elems.append(ax.text(x, y+0.1, s, fontsize=8))

    @classmethod
    def policy(cls, pose: np.ndarray, goal: Goal) -> Tuple[float,float]:
        raise NotImplementedError()

        
class FastSlam2Agent(EstimationAgent):
    def __init__(self, 
                 time_interval: float, 
                 nu: float, 
                 omega: float, 
                 estimator: Estimator):
        super().__init__(time_interval, nu, omega, estimator)

    def decision(self, observation: Optional[np.ndarray] = None) -> Tuple[float,float]:
        """ロボットの自己位置を
        [1] 状態方程式で更新
        [2] 観測方程式で補正
        """

        # 状態方程式
        self.estimator.motion_update(
            self.prev_nu, 
            self.prev_omega, 
            self.time_interval, 
            observation # センサー情報を追加
        )

        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        return self.nu, self.omega

    def draw(self, ax, elems):
        raise NotImplementedError()

    @classmethod
    def policy(cls, pose: np.ndarray, goal: Goal) -> Tuple[float,float]:
        raise NotImplementedError()

    
class LoggerAgent(Agent):
    def __init__(self, 
                 nu: float, 
                 omega: float, 
                 interval_time: float, 
                 init_pose: np.ndarray):
        
        super().__init__(nu, omega)

        # 更新時間と初期姿勢を変数に加える
        self.interval_time: float = interval_time
        self.pose: np.ndarray = init_pose
        self.step: int = 0
        current_dir: str = Path(__file__).resolve().parent
        self.log: str = open(os.path.join(str(current_dir), "graph_slam_log_1.txt"), "w")

    def decision(self, observation: Optional[np.ndarray] = None) -> Tuple[float,float]:
        if len(observation) != 0: # ランドマークが観測されていない姿勢は記録しない
            self.log.write("x {} {} {} {}\n".format(self.step, *self.pose))
            for obs in observation:
                # z : step phi Zx, Zy, ZΘ -> カメラとランドマークの相対角度 + (ランドマークの姿勢)
                self.log.write("z {} {} {} {} {}\n".format(self.step, obs[1], *obs[0]))

            self.step += 1
            self.log.flush()

        self.pose = IdealRobot.state_transition(self.nu,
                                                self.omega,
                                                self.interval_time,
                                                self.pose)
        return self.nu, self.omega

    def draw(self, ax, elems):
        raise NotImplementedError()

    @classmethod
    def policy(cls, pose: np.ndarray, goal: Goal):
        raise NotImplementedError()



# 強化学習エージェント＠固定方策(リスクである水たまりを突っ切る行動選択)
class PuddleIgnoreAgent(EstimationAgent):
    def __init__(self, 
                 time_interval: float, 
                 nu: float, 
                 omega: float, 
                 estimator: float, 
                 goal: Optional[Goal], 
                 puddle_coef: float = 100): 
        super().__init__(time_interval, nu, omega, estimator)

        self.puddle_coef: float = puddle_coef
        self.puddle_depth: float = 0.0
        self.total_reward: float = 0.0
        self.in_goal: bool = False
        self.final_value: float = 0.0
        self.goal: Optional[Goal] = goal

    def reward_per_sec(self) -> float:
        return -1.0 - self.puddle_depth * self.puddle_coef

    @classmethod
    def policy(cls, pose: np.ndarray, goal: Goal) -> Tuple[float,float]:
        """水たまりを無視した固定方策(初期位置からゴールまで一直線に進む)"""
        x, y, theta = pose
        dx, dy = goal.pos[0]-x, goal.pos[1]-y

        # ゴールの方向(degreeに変換)
        direction = int((math.atan2(dy, dx) - theta)*180/math.pi)
        direction = (direction + 360*1000 + 180) % 360 - 180 # 方角を-180 ~ +180[deg]に正規化. ロボットが-1000回転すると破綻.
        # print(f"direction: {direction}[deg] @policy")

        if direction > 10: nu, omega = 0.0, 2.0
        elif direction < -10: nu, omega = 0.0, -2.0
        else: nu, omega = 1.0, 0.0

        # print(f"nu: {nu:.1f}[m/s] omega: {omega:.1f}[rad/s] @policy")
        
        return nu, omega

    def decision(self, observation: Optional[np.ndarray] = None) -> Tuple[float,float]:
        if self.in_goal:
            return 0.0, 0.0

        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.estimator.observation_update(observation)

        self.total_reward += self.time_interval * self.reward_per_sec() # Δtにおける報酬

        nu, omega = self.policy(self.estimator.pose, self.goal)
        self.prev_nu, self.prev_omega = nu, omega
        return nu, omega

    def draw(self, ax, elems):
        super().draw(ax, elems)
        x, y, _ = self.estimator.pose
        elems.append(ax.text(x+1.0, y-0.5, "reward/sec:" + str(self.reward_per_sec()), fontsize=8))
        J = self.total_reward+self.final_value # 評価値J
        elems.append(ax.text(x+1.0, y-1.0, "evaluation: {:.1f}".format(J), fontsize=8))

    

    
class DpPolicyAgent(PuddleIgnoreAgent):
    """強化学習エージェント@動的計画法によって取得した方策で行動する"""
    def __init__(self, 
                    time_interval: float,
                    estimator: Estimator,
                    goal: Optional[Goal],
                    puddle_coef: float = 100,
                    widths: np.ndarray = np.array([0.2,0.2,math.pi/18]).T,
                    lowerleft: np.ndarray = np.array([-4,-4]).T,
                    upperright: np.ndarray = np.array([4,4]).T,
                    policy_filename: Optional[str] = None,
                    disable_init_policy: bool = False,
                    ):

        super().__init__(time_interval, 
                         nu=0, omega=0, 
                         estimator=estimator, 
                         goal=goal, 
                         puddle_coef=puddle_coef)

        self.pose_min: np.ndarray = np.r_[lowerleft, 0]
        self.pose_max: np.ndarray = np.r_[upperright, 2*math.pi]
        self.widths: np.ndarray = widths
        self.index_nums: np.ndarray = ((self.pose_max-self.pose_min)/self.widths).astype(int)

        self._disable_init_policy: bool = disable_init_policy
        self.policy_filename: str = 'dp_policy.txt'
        if policy_filename: 
            self.policy_filename = policy_filename
        print(self.policy_filename)
        self.policy_data: np.ndarray = self.init_policy(self.index_nums)
        
        
    # def __getattribute__(self, name):
    #     # _disable_init_policyがTrue、かつ無効化対象のメソッド名の場合は存在しないことにする
    #     if name == 'policy' and object.__getattribute__(self, '_disable_init_policy'):
    #         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    #     return super().__getattribute__(name)

    def init_policy(self, index_nums: np.ndarray) -> np.ndarray:
        tmp = np.zeros(np.r_[index_nums, 2]) # 制御指令(制御速度, 制御角速度)を追加. 計5次元

        with open(os.sep.join([os.getcwd(), self.policy_filename]), mode='r') as f:
            """ファイル形式
            sx:3 sy:10 st:4 uv:0.0 uw:-2.0
            sx:3 sy:10 st:5 uv:0.0 uw:-2.0
            sx:3 sy:10 st:6 uv:0.0 uw:-2.0
            sx:3 sy:10 st:7 uv:0.0 uw:-2.0
            """
            for line in f.readlines():
                items = line.split()
                sx = int(items[0].split(':')[-1]) # 状態(X軸)
                sy = int(items[1].split(':')[-1]) # 状態(Y軸)
                st = int(items[2].split(':')[-1]) # 状態(Z軸)
                uv = float(items[3].split(':')[-1]) # 制御指令(速度v)
                uw = float(items[4].split(':')[-1]) # 制御指令(回転速度w)

                tmp[sx,sy,st] = [uv,uw]

        return tmp

    def to_index(self, pose: np.ndarray, pose_min: np.ndarray, index_nums: np.ndarray, widths: np.ndarray):
        """姿勢状態をインデックスに正規化して離散状態に変換."""

        # 姿勢をインデックスに変換
        index = np.floor((pose - pose_min) / widths).astype(int)

        index[2] = (index[2] + index_nums[2]*1000) % index_nums[2] # 角度の正規化
        for i in [0,1]: # 端の処理。内側の座標の方策を使う
            if index[i] < 0: index[i] = 0
            elif index[i] >= index_nums[i]: index[i] = index_nums[i] - 1

        return tuple(index) # ベクトルのままだとインデックスに使えないのでタプル化

    @override
    def policy(self, pose: np.ndarray, goal=Optional[Goal]):
        """姿勢から離散状態のインデックスを作って方策を参照して返す
        self._disable_init_policy = Trueの場合は、親クラスPuddleIgnoreAgentの固定方策を利用する.
        """
        if self._disable_init_policy:
            return PuddleIgnoreAgent.policy(pose, goal)
        
        return self.policy_data[self.to_index(pose, self.pose_min, self.index_nums, self.widths)]


class QAgent(DpPolicyAgent):
    """強化学習@Q学習のエージェント"""
    def __init__(self,
                 time_interval: float,
                 estimator: Estimator,
                 goal: Optional[Goal] = None,
                 puddle_coef: float = 100,
                 widths: np.ndarray = np.array([0.2,0.2,math.pi/18]).T,
                 lowerleft: np.ndarray = np.array([-4,-4]).T,
                 upperright: np.ndarray = np.array([4,4]).T,
                 dev_borders: list[int] = [0.1, 0.2, 0.4, 0.8],
                 policy_filename: Optional[str] = None,
                 value_filename: Optional[str] = None,
                 disable_init_policy: bool = False,
                 ):
        super().__init__(time_interval, 
                         estimator, 
                         goal, 
                         puddle_coef, 
                         widths, 
                         lowerleft, 
                         upperright,
                         policy_filename,
                         disable_init_policy)

        # 環境の離散状態 s ∈ S
        # 連続の状態空間をwidths=np.array([0.2,0.2,math.pi/180]).T
        # で分割したインデックス集合
        nx,ny,nt = self.index_nums # 状態空間の各軸の分割数 (X軸,Y軸,回転軸)
        self.indexes = list(itertools.product(range(nx),range(ny),range(nt)))

        # 離散状態の一つに対応する行動
        self.actions = list(set([tuple(self.policy_data[i]) for i in self.indexes]))

        self.policy_filename: str | None = policy_filename
        self.valud_filename: str | None = value_filename

        # 初期値の状態行動対(s,a)と行動価値Qの読み込み
        if value_filename:
            self.ss = self.set_action_value_function(value_filename)


    def set_action_value_function(self, value_filename: str):
        ss = {} # State Space { (0,1,2): StateInfo, .... }

        with open(value_filename, mode='r') as f:
            """ファイルの形式
            ....
            sx:0 sy:0 st:5 V:-1.5331009792117058
            sx:0 sy:0 st:6 V:-1.6080748791905841
            sx:0 sy:0 st:7 V:-1.6930800991948085
            sx:0 sy:0 st:8 V:-1.7760790551939638
            ....
            
            """
            for line in f.readlines():
                items = line.split()
                sx = int(items[0].split(':')[-1]) # 状態(X軸)
                sy = int(items[1].split(':')[-1]) # 状態(Y軸)
                st = int(items[2].split(':')[-1]) # 状態(Z軸)
                V = float(items[3].split(':')[-1])  # 状態価値
                index, value = (sx,sy,st), V

                # StateInfoオブジェクトを割り当てて初期化
                ss[index] = StateInfo(len(self.actions))

                # 行動価値の初期化
                for i, a in enumerate(self.actions):
                    # 方策と一致しない場合は, ファイルの行動価値の値から少し引く.
                    ss[index].q[i] = (value if 
                                      tuple(self.policy_data[index]) == a 
                                      else value - 0.1)
        return ss

    @override
    def policy(self, pose: np.ndarray, goal=Optional[Goal]):
        # 状態行動対(s,a)に行動価値Qを割り当てていない場合, 固定の方策を実行
        if self.valud_filename is None:
            return super().policy(pose, goal)

        # 連続な状態変数から特定の離散状態を特定する
        index = self.to_index(pose, self.pose_min, 
                              self.index_nums, self.widths)

        # 行動価値関数を使って行動決定
        a = self.ss[tuple(index)].pi()
        return self.actions[a]

        


class StateInfo:
    def __init__(self, action_num: int, epsilon: float = 0.3):
        self.q = np.zeros(action_num) # 0軸: 状態, 1軸: 行動
        self.epsilon: float = epsilon

    def greedy(self):
        return np.argmax(self.q)

    def epsilon_greedy(self, epsilon: float):
        if random.random() < epsilon:
            return random.choice(range(len(self.q)))
        else:
            return self.greedy()

    def pi(self):
        """ε-グリーディ化した方策 π(a|s)"""
        return self.epsilon_greedy(self.epsilon)


    

    
