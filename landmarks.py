import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as anm


class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x,y]).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))