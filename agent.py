import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as anm


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega