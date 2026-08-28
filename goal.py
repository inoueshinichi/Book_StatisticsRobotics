"""強化学習におけるゴール
"""
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

class Goal:
    def __init__(self, x, y, radius=0.3, value=0.0):
        self.pos = np.array([x, y]).T
        self.radius = radius
        self.value = value

    def draw(self, ax, elems):
        x,y = self.pos
        c = ax.scatter(x + 0.16, y + 0.5, s=50, marker='>', label='landmarks', color='red') # 赤旗
        elems.append(c)
        elems += ax.plot([x,x], [y,y+0.6], color='black') # 旗棒

    def inside(self, pose):
        return self.radius > math.sqrt((self.pos[0]-pose[0])**2 + (self.pos[1]-pose[1])**2)

    
