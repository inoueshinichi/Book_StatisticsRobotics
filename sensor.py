import math
from typing import List, Dict, Tuple, Union, Optional
from abc import ABC, ABCMeta, abstractmethod

import numpy as np
from scipy.stats import expon, norm, uniform

from map import Map

class Sensor(ABC):
    @abstractmethod
    def visible(self, obj_pose: Optional[np.ndarray]) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def data(self, sensor_pose: np.ndarray) -> List[Tuple[np.ndarray,int]]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def observation_function(cls, sensor_pose: np.ndarray, obj_pose: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class IdealCamera(Sensor):
    def __init__(self, 
                 envmap: Map,
                 distance_range: Tuple[float,float] = (0.5,6.0),
                 direction_range: Tuple[float,float] = (-math.pi/3,math.pi/3),
                 ):
        self.map: Map = envmap
        self.lastdata: List[Tuple[np.ndarray,int]] = []
    
        self.distance_range: Tuple[float,float] = distance_range
        self.direction_range: Tuple[float,float] = direction_range

    def visible(self, obj_pose: Optional[np.ndarray]) -> bool:
        if obj_pose is None:
            return False
        
        return self.distance_range[0] <= obj_pose[0] <= self.distance_range[1] \
            and self.direction_range[0] <= obj_pose[1] <= self.direction_range[1]

    def data(self, sensor_pose: np.ndarray, 
             orientation_noise: float = 0) -> List[Tuple[np.ndarray,int]]:
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(sensor_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id))
            
        self.lastdata = observed
        return observed
    
    # センサからのデータ取得(観測方程式)
    @classmethod
    def observation_function(cls, sensor_pose, obj_pose) -> np.ndarray:
        diff = obj_pose - sensor_pose[:2]
        phi = math.atan2(diff[1], diff[0]) - sensor_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi
        return np.array([np.hypot(*diff), phi]).T

    def draw(self, ax, elems, sensor_pose: np.ndarray):
        for lm in self.lastdata:
            x, y, theta = sensor_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x,lx], [y,ly], color="pink")
        
    

class Camera(IdealCamera):
    def __init__(self, 
                 envmap: Map,
                 distance_range: Tuple[float,float] = (0.5,6.0),
                 direction_range: Tuple[float,float] = (-math.pi/3,math.pi/3),
                 distance_noise_rate: float = 0.1,    # 距離に加える雑音の標準偏差の割合
                 direction_noise: float = math.pi/90, # 方角に加える雑音の標準偏差
                 distance_bias_rate_stddev: float =0.1,    # 距離に加えるバイアス
                 direction_bias_stddev: float = math.pi/90, # 方角に加えるバイアス
                 phantom_prob: float = 0.0,       # ファントムの出現確率
                 phantom_range_x: Tuple[float,float] = (-5,5), # ファントムの出現範囲(x軸)
                 phantom_range_y: Tuple[float,float] = (-5,5), # ファントムの出現範囲(y軸)
                 oversight_prob: float = 0.1, # 観測値の見落とし確率
                 occlusion_prob: float = 0.0, # 観測値のオクルージョン発生確率
                 ):
        
        super().__init__(envmap, distance_range, direction_range)

        # ノイズ
        self.distance_noise_rate: float = distance_noise_rate
        self.direction_noise: float = direction_noise

        # バイアス
        self.distance_bias_rate_stddev: float = distance_bias_rate_stddev
        self.direction_bias_stddev: float = direction_bias_stddev
        self.distance_bias_rate_std: Union[np.ndarray, float] = norm.rvs(scale=self.distance_bias_rate_stddev)
        self.direction_bias: Union[np.ndarray, float] = norm.rvs(scale=self.direction_bias_stddev)

        # ファントムの出現範囲と出現確率
        rx: float 
        ry: float
        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc=(rx[0],ry[0]), scale=(rx[1]-rx[0],ry[1]-ry[0]))
        self.phantom_prob: float = phantom_prob

        # 観測値の見落とし確率
        self.oversight_prob: float = oversight_prob

        # 観測値のオクルージョン発生率
        self.occlusion_prob: float = occlusion_prob

    def noise(self, relpos: np.ndarray) -> np.ndarray:
        """確率で発生するセンサノイズ"""
        ell = norm.rvs(loc=relpos[0], scale=relpos[0]*self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise)
        return np.array([ell,phi]).T
    
    def bias(self, relpos: np.ndarray) -> np.ndarray:
        """確率で発生するセンサバイアス"""
        return relpos + np.array([relpos[0]*self.distance_bias_rate_std,
                                  self.direction_bias]).T
    
    def phantom(self, sensor_pose: np.ndarray, relpos: np.ndarray) -> np.ndarray:
        """確率で発生するファントムの影響"""
        if uniform.rvs() < self.phantom_prob:
            pos = np.array(self.phantom_dist.rvs()).T # (lm_x,lm_y)
            return self.observation_function(sensor_pose, pos)
        else:
            return relpos
        
    def oversight(self, relpos: np.ndarray) -> Optional[np.ndarray]:
        """センサーの測定範囲内か?"""
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpos
        
    def occlusion(self, relpos: np.ndarray) -> np.ndarray:
        """確率で発生するオクルージョン発生による影響"""
        if uniform.rvs() < self.occlusion_prob:
            ell = relpos[0] + uniform.rvs() * (self.distance_range[1] - relpos[0])
            phi = relpos[1]
            return np.array([ell, phi]).T
        else:
            return relpos
    
    def data(self, sensor_pose: np.ndarray, 
             orientation_noise: float = 0) -> List[Tuple[np.ndarray,int]]:
        """カメラによるランドマークの観測データの取得"""
        observed: List[Tuple[np.ndarray,int]] = []
        for lm in self.map.landmarks:
            z = self.observation_function(sensor_pose, lm.pos)
            z = self.phantom(sensor_pose, z) # 観測値を上書きしてファントムを出現させる
            z = self.occlusion(z) # オクルージョン(観測したが観測値が真値からおおきく離れた値を取得する)
            z = self.oversight(z) # 見落とし
            if self.visible(z):
                z = self.bias(z)
                z = self.noise(z)
                observed.append((z,lm.id))
        
        self.lastdata = observed
        return observed
    
class PsiCamera(Camera):

    def data(self, 
             sensor_pose: np.ndarray, 
             orientation_noise: float = math.pi/90) -> List[Tuple[np.ndarray,int]]:
        # orientation_noiseを追加。psiの雑音の大きさをセット
        observed: List[Tuple[np.ndarray,int]] = []

        for lm in self.map.landmarks:
            # ランドマークとカメラの姿勢差
            psi = norm.rvs(loc=math.atan2(sensor_pose[1]-lm.pos[1], 
                                          sensor_pose[0]-lm.pos[0]),
                                          scale=orientation_noise)
            # 観測値(Xx, Xy, XΘ, Zx,Zy, ZΘ) -> 極座標(Zl, Zφ)
            z = self.observation_function(sensor_pose, lm.pos)
            z = self.phantom(sensor_pose, z)
            z = self.oversight(z)
            if self.visible(z): # FOVの範囲内
                z = self.bias(z)
                z = self.noise(z)
                observed.append(([z[0], z[1], psi], lm.id))

            self.lastdata = observed

        return observed
    



