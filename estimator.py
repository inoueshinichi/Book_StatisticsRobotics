"""ロボットの自己位置推定器
・カルマンフィルタ
・パーティクルフィルタ (MCL)
・パーティクルフィルタ (KLD Sampling MCL)
"""
import math
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, Tuple, List, Optional, Union
import numpy as np

from map import Map
from kalman_filter import KalmanFilter, GlobalKalmanFilter
from mcl import Mcl, GlobalMcl, ResetMcl
from kld_mcl import KldMcl, GlobalKldMcl
from particles import Particle

class Estimator(ABC):
    @abstractmethod
    def motion_update(self, nu: float, omega: float, time: float, 
                      observation: Optional[np.ndarray] = None):
        raise NotImplementedError()

    @abstractmethod
    def observation_update(self, observation):
        raise NotImplementedError()

    @abstractmethod
    def draw(self, ax, elems):
        raise NotImplementedError()


class KalmanFilterEstimator(Estimator):
    def __init__(self,
                 envmap: Map, # 環境地図
                 init_pose: np.ndarray, # ロボットの初期位置
                 motion_noise_stds: Dict[str, float] = {
                     # ばらつき(白色雑音)
                     "nn": 0.19, "no": 0.001, "on": 0.13, "oo": 0.2,
                 },
                 distance_dev_rate: float = 0.14, # 直線成分のばらつき
                 direction_dev: float = 0.05, # 方角成分のばらつき
                 ):
        super().__init__()

        # 推定手法
        self.kf = KalmanFilter(
            envmap=envmap, 
            init_pose=init_pose, 
            motion_noise_stds=motion_noise_stds,
            distance_dev_rate=distance_dev_rate,
            direction_dev=direction_dev,
        )

    def motion_update(self, nu: float, omega: float, deltatime: float):
        self.kf.motion_update(nu, omega, deltatime)

    def observation_update(self, observation):
        self.kf.observation_update(observation)

    def draw(self, ax, elems):
        self.kf.draw(ax, elems)

    @property
    def pose(self) -> np.ndarray:
        return self.kf.pose


class GlobalKalmanFilterEstimator(Estimator):
    def __init__(self,
                 envmap: Map, # 環境地図
                 motion_noise_stds:Dict[str, float] = {
                    # ばらつき(白色雑音)
                    "nn": 0.19, "no": 0.001, "on": 0.13, "oo": 0.2,
                 },
                 distance_dev_rate: float = 0.14, # 直線成分のばらつき
                 direction_dev: float = 0.05, # 方角成分のばらつき
                 ):
        super().__init__()

        # 推定手法
        self.gkf = GlobalKalmanFilter(
            envmap=envmap, 
            motion_noise_stds=motion_noise_stds,
            distance_dev_rate=distance_dev_rate,
            direction_dev=direction_dev,
        )

    def motion_update(self, nu: float, omega: float, deltatime: float):
        self.gkf.motion_update(nu, omega, deltatime)

    def observation_update(self, observation):
        self.gkf.observation_update(observation)

    def draw(self, ax, elems):
        self.gkf.draw(ax, elems)

    @property
    def pose(self) -> np.ndarray:
        return self.gkf.pose


class MclParticleFilterEstimator(Estimator):
    def __init__(self,
                 envmap: Map, # 環境地図
                 init_pose: np.ndarray, # ロボットの初期位置
                 num: int, # パーティクル数
                 motion_noise_stds:Dict[str, float] = {
                    # ばらつき(白色雑音)
                    "nn": 0.19, "no": 0.001, "on": 0.13, "oo": 0.2,
                 },
                 distance_dev_rate: float = 0.14, # 直線成分のばらつき
                 direction_dev: float = 0.05, # 方角成分のばらつき
                 ):
        super().__init__()

        # 推定手法
        self.mcl = Mcl(
            envmap=envmap, 
            init_pose=init_pose,
            num=num,
            motion_noise_stds=motion_noise_stds,
            distance_dev_rate=distance_dev_rate,
            direction_dev=direction_dev,
        )

    def motion_update(self, nu: float, omega: float, deltatime: float):
            self.mcl.motion_update(nu, omega, deltatime)
    
    def observation_update(self, observation):
        self.mcl.observation_update(observation)

    def draw(self, ax, elems):
        # print(f"estimator draw")
        self.mcl.draw(ax, elems)

    @property
    def pose(self) -> np.ndarray:
        return self.mcl.pose

    @property
    def perticles(self) -> List[Particle]:
        return self.gmcl.particles


class GlobalMclParticleFilterEstimator(Estimator):
    def __init__(self,
                 envmap: Map, # 環境地図
                 num: int, # パーティクル数
                 motion_noise_stds:Dict[str, float] = {
                    # ばらつき(白色雑音)
                    "nn": 0.19, "no": 0.001, "on": 0.13, "oo": 0.2,
                 },
                 distance_dev_rate: float = 0.14, # 直線成分のばらつき
                 direction_dev: float = 0.05, # 方角成分のばらつき
                 ):
        super().__init__()

        # 推定手法
        self.gmcl = GlobalMcl(
            envmap=envmap,
            num=num,
            motion_noise_stds=motion_noise_stds,
            distance_dev_rate=distance_dev_rate,
            direction_dev=direction_dev,
        )

    def motion_update(self, nu: float, omega: float, deltatime: float):
            self.gmcl.motion_update(nu, omega, deltatime)
    
    def observation_update(self, observation):
        self.gmcl.observation_update(observation)

    def draw(self, ax, elems):
        self.gmcl.draw(ax, elems)

    @property
    def pose(self) -> np.ndarray:
        return self.gmcl.pose

    @property
    def perticles(self) -> List[Particle]:
        return self.gmcl.particles


class ResetMclParticleFilterEstimator(Estimator):
    def __init__(self,
                 envmap: Map, # 環境地図
                 init_pose: np.ndarray, # 初期位置
                 num: int, # パーティクル数
                 motion_noise_stds:Dict[str, float] = {
                    # ばらつき(白色雑音)
                    "nn": 0.19, "no": 0.001, "on": 0.13, "oo": 0.2,
                 },
                 distance_dev_rate: float = 0.14, # 直線成分のばらつき
                 direction_dev: float = 0.05, # 方角成分のばらつき
                ):

        super().__init__()

        # 推定手法
        self.remcl = ResetMcl(
            envmap=envmap,
            init_pose=init_pose,
            num=num,
            motion_noise_stds=motion_noise_stds,
            distance_dev_rate=distance_dev_rate,
            direction_dev=direction_dev,
        )

    def motion_update(self, nu: float, omega: float, deltatime: float):
            self.remcl.motion_update(nu, omega, deltatime)
    
    def observation_update(self, observation):
        self.remcl.observation_update(observation)

    def draw(self, ax, elems):
        self.remcl.draw(ax, elems)

    @property
    def pose(self) -> np.ndarray:
        return self.remcl.pose

    @property
    def alphas(self) -> Dict:
        return self.remcl.alphas

    @property
    def perticles(self) -> List[Particle]:
        return self.remcl.particles

    
class KldMclParticleFilterEstimator(Estimator):
    def __init__(self,
                 envmap: Map, # 環境地図
                 init_pose: np.ndarray, # 初期位置
                 max_num: int, # 最大生成パーティクル数
                 motion_noise_stds:Dict[str, float] = {
                    # ばらつき(白色雑音)
                    "nn": 0.19, "no": 0.001, "on": 0.13, "oo": 0.2,
                 },
                 distance_dev_rate: float = 0.14, # 直線成分のばらつき
                 direction_dev: float = 0.05, # 方角成分のばらつき
                 widths: np.ndarray = np.array([0.2, 0.2, math.pi/18]).T,
                 epsilon: float = 0.1,
                 delta: float = 0.01,
                 ):
    
         super().__init__()

         # 推定手法
         self.kldmcl = KldMcl(
            envmap=envmap,
            init_pose=init_pose,
            max_num=max_num,
            motion_noise_stds=motion_noise_stds,
            distance_dev_rate=distance_dev_rate,
            direction_dev=direction_dev,
            widths=widths,
            epsilon=epsilon,
            delta=delta,
         )

    def motion_update(self, nu: float, omega: float, deltatime: float):
        self.kldmcl.motion_update(nu, omega, deltatime)
    
    def observation_update(self, observation):
        self.kldmcl.observation_update(observation)

    def draw(self, ax, elems):
        self.kldmcl.draw(ax, elems)

    @property
    def pose(self) -> np.ndarray:
        return self.kldmcl.pose


class GlobalKldMclParticleFilterEstimator(Estimator):
    def __init__(self,
                 envmap: Map, # 環境地図
                 max_num: int, # 最大生成パーティクル数
                 motion_noise_stds: Dict[str, float] = {
                     # ばらつき(白色雑音)
                     "nn": 0.19, "no": 0.001, "on": 0.13, "oo": 0.2,
                 },
                 distance_dev_rate: float = 0.14, # 直線成分のばらつき
                 direction_dev: float = 0.05, # 方角成分のばらつき
                 ):

        super().__init__()

        # 推定手法
        self.gkldmcl = GlobalKldMcl(
            envmap=envmap,
            max_num=max_num,
            motion_noise_stds=motion_noise_stds,
            distance_dev_rate=distance_dev_rate,
            direction_dev=direction_dev,
        )

    def motion_update(self, nu: float, omega: float, deltatime: float):
        self.gkldmcl.motion_update(nu, omega, deltatime)
        
    def observation_update(self, observation):
        self.gkldmcl.observation_update(observation)

    def draw(self, ax, elems):
        self.gkldmcl.draw(ax, elems)

    @property
    def pose(self) -> np.ndarray:
        return self.gkldmcl.pose

