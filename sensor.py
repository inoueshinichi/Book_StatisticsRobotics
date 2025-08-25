import math
import numpy as np
from scipy.stats import expon, norm, uniform

class IdealCamera:
    def __init__(self, 
                 env_map,
                 distance_range=(0.5,6.0),
                 direction_range=(-math.pi/3,math.pi/3),
                 ):
        self.map = env_map
        self.lastdata = []
    
        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, polarpos):
        if polarpos is None:
            return False
        
        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1] \
            and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id))
            
        self.lastdata = observed
        return observed
    
    # センサからのデータ取得(観測方程式)
    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi
        return np.array([np.hypot(*diff), phi]).T

    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x,lx], [y,ly], color="pink")
        
    

class Camera(IdealCamera):
    def __init__(self, 
                 env_map,
                 distance_range=(0.5,6.0),
                 direction_range=(-math.pi/3,math.pi/3),
                 distance_noise_rate=0.1,    # 距離に加える雑音の標準偏差の割合
                 direction_noise=math.pi/90, # 方角に加える雑音の標準偏差
                 distance_bias_rate_stddev=0.1,    # 距離に加えるバイアス
                 direction_bias_stddev=math.pi/90, # 方角に加えるバイアス
                 phantom_prob=0.0,       # ファントムの出現確率
                 phantom_range_x=(-5,5), # ファントムの出現範囲(x軸)
                 phantom_range_y=(-5,5), # ファントムの出現範囲(y軸)
                 oversight_prob=0.1, # 観測値の見落とし確率
                 occlusion_prob=0.0, # 観測値のオクルージョン発生確率
                 ):
        
        super().__init__(env_map, distance_range, direction_range)

        # ノイズ
        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise

        # バイアス
        self.distance_bias_rate_stddev = distance_bias_rate_stddev
        self.direction_bias_stddev = direction_bias_stddev
        self.distance_bias_rate_std = norm.rvs(scale=self.distance_bias_rate_stddev)
        self.direction_bias = norm.rvs(scale=self.direction_bias_stddev)

        # ファントム
        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc=(rx[0],ry[0]), scale=(rx[1]-rx[0],ry[1]-ry[0]))
        self.phantom_prob = phantom_prob

        # 見落とし
        self.oversight_prob = oversight_prob

        # オクルージョン
        self.occlusion_prob = occlusion_prob

    def noise(self, relpos):
        ell = norm.rvs(loc=relpos[0], scale=relpos[0]*self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise)
        return np.array([ell,phi]).T
    
    def bias(self, relpos):
        return relpos + np.array([relpos[0]*self.distance_bias_rate_std,
                                  self.direction_bias]).T
    
    def phantom(self, cam_pose, relpos):
        if uniform.rvs() < self.phantom_prob:
            pos = np.array(self.phantom_dist.rvs()).T # (lm_x,lm_y)
            return self.observation_function(cam_pose, pos)
        else:
            return relpos
        
    def oversight(self, relpos):
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpos
        
    def occlusion(self, relpos):
        if uniform.rvs() < self.occlusion_prob:
            ell = relpos[0] + uniform.rvs() * (self.distance_range[1] - relpos[0])
            phi = relpos[1]
            return np.array([ell, phi]).T
        else:
            return relpos
    
    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            z = self.phantom(cam_pose, z) # 観測値を上書きしてファントムを出現させる
            z = self.occlusion(z) # オクルージョン(観測したが観測値が真値からおおきく離れた値を取得する)
            z = self.oversight(z) # 見落とし
            if self.visible(z):
                z = self.bias(z)
                z = self.noise(z)
                observed.append((z,lm.id))
        
        self.lastdata = observed
        return observed
    


