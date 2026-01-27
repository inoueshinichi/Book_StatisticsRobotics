import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as anm

from kalman_filter import sigma_ellipse

class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x,y]).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))


# 地図に指定された個数だけ、ランドマーク位置推定用のガウス分布を表す
class EstimatedLandmark(Landmark):
    def __init__(self):
        super().__init__(0, 0) # 姿勢を元のクラスのposeに設定
        # self.cov = np.array([[1,0],[0,2]]) # 描画のテスト用の値。あとでNoneにする
        self.cov = None

    def draw(self, ax, elems):
        # 共分散が定義されていない場合描画なし
        if self.cov is None:
            return
        
        # 推定位置に青い星を置く
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="blue")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))

        # 誤差楕円を書く
        e = sigma_ellipse(self.pos, self.cov, 3)
        elems.append(ax.add_patch(e))

    