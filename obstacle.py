"""障害物
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


# 水たまり(深さで深刻度を定義)
class Puddle:
    def __init__(self, lowerleft, upperright, depth):
        self.lowerleft = lowerleft
        self.upperright = upperright
        self.depth = depth

    def draw(self, ax, elems):
        w = self.upperright[0] - self.lowerleft[0]
        h = self.upperright[1] - self.lowerleft[1]
        r = patches.Rectangle(self.lowerleft, w, h, color='blue', alpha=self.depth)
        elems.append(ax.add_patch(r))

    def inside(self, pose):
        return all([self.lowerleft[i] < pose[i] < self.upperright[i] for i in [0, 1]])