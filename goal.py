"""強化学習におけるゴール
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

class Goal:
    def __init__(self, x, y, radius = 0.3):
        self.pos = np.array([x, y]).T
        self.radius = radius

    def draw(self, ax, elems):
        x,y = self.pos
        c = ax.scatter(x + 0.16, y + 0.5, s=50, marker='>', label='landmarks', color='red') # 赤旗
        elems.append(c)
        elems += ax.plot([x,x], [y,y+0.6], color='black') # 旗棒


