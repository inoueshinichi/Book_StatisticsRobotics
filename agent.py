import os
from pathlib import Path
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Set, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import math
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
        """方策"""
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
                    goal: Goal,
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
        if policy_filename: self.policy_filename = policy_filename
        print(self.policy_filename)
        self.policy_data: np.ndarray = self.init_policy(self.index_nums)
        
        
    # def __getattribute__(self, name):
    #     # _disable_init_policyがTrue、かつ無効化対象のメソッド名の場合は存在しないことにする
    #     if name == 'policy' and object.__getattribute__(self, '_disable_init_policy'):
    #         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    #     return super().__getattribute__(name)

    def init_policy(self, index_nums: np.ndarray) -> np.ndarray:
        tmp = np.zeros(np.r_[index_nums, 2]) # 制御指令(制御速度, 制御角速度)を追加. 計5次元
        for line in open(os.sep.join([os.getcwd(), self.policy_filename]), 'r'):
            d = line.split()
            tmp[int(d[0]), int(d[1]), int(d[2])] = [float(d[3]),float(d[4])]

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
                 goal: Goal,
                 puddle_coef: float = 100,
                 widths: np.ndarray = np.array([0.2,0.2,math.pi/18]).T,
                 lowerleft: np.ndarray = np.array([-4,-4]).T,
                 upperright: np.ndarray = np.array([4,4]).T,
                 policy_filename: Optional[str] = None,
                 disable_init_policy: bool = False,
                 ):
        super().__init__(time_interval, estimator, goal, puddle_coef, 
                         widths, lowerleft, upperright,
                         policy_filename, disable_init_policy)

