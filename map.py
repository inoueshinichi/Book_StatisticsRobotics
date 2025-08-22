import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as anm


class Map:
    def __init__(self):
        self.landmarks = []

    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    def draw(self, ax, elems):
        for lm in self.landmarks:
            lm.draw(ax, elems)