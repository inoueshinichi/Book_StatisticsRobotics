import math
import numpy as np


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
                 ):
        
        super().__init__(env_map, distance_range, direction_range)

        # ノイズ
        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise

