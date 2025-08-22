import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as anm


class World:
    def __init__(self, 
                 time_span: float,     # シミュレーションタイムduration
                 time_interval: float, # 1コマの時間Δt
                 debug=False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval

    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(4,4)) # 4x4 inch図
        ax = fig.add_subplot(111)
        ax.set_aspect('equal') # 縦横比を座標の値と一致させる
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)

        elems = []

        if self.debug:
            for i in range(1000): 
                self.one_step(i, elems, ax) # デバッグ時はアニメーションさせない
        else:
            print('Start animation')
            
            # シミュレーション時間とタイムスパンからアニメーションを作成
            self.ani = anm.FuncAnimation(fig,
                                         self.one_step,
                                         fargs=(elems, ax),
                                         frames=int(self.time_span /self.time_interval ) + 1,
                                         interval=int(self.time_interval * 1000),
                                         repeat=False)
            
            print('objects', self.objects)
            print('elems ', elems)

            plt.show()


    def one_step(self, 
                 i,     # ステップ番号 
                 elems, # 描画リスト
                 ax,    # サブプロット
                 ):
        # アニメーションを1コマ進めるメソッド
        while elems: 
            elems.pop().remove() # 二重描画を防止
        time_str = "t = %.2f[s]" % (self.time_interval * i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)