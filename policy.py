
import math
import numpy as np
import itertools
import collections
from typing import Dict, OrderedDict, List, Tuple, Set, Union, Optional


from world import PuddleWorld, Goal, Puddle
from agent import PuddleIgnoreAgent
from robot import IdealRobot


class PolicyEvaluator:
    def __init__(self, 
                 widths, 
                 goal, 
                 puddles, 
                 time_interval, 
                 sampling_num, 
                 puddle_coef=100,
                 lowerleft=np.array([-4,-4]).T, 
                 upperright=np.array([4,4]).T):
        self.pose_min = np.r_[lowerleft, 0] # 連結(min_x, min_y, min_theta)
        self.pose_max = np.r_[upperright, math.pi*2] # theta: 0 ~ 2pi
        self.widths = widths
        self.goal = goal
        

        # 状態空間の離散化(インデックス)
        self.index_nums = ((self.pose_max - self.pose_min) / self.widths).astype(int)
        nx, ny, nt = self.index_nums
        self.indexes = list(itertools.product(range(nx), range(ny), range(nt))) # 全部のインデックスの組み合わせを作る

        # 状態価値Vπ
        self.value_function, self.final_state_flags = self.init_value_function()

        # 方策
        self.policy = self.init_policy() # [velocity,omega] の集合

        # 行動
        self.actions = list(set([tuple(self.policy[i].tolist()) for i in self.indexes])) # ユニークな行動を取得
        self.state_transition_probs = self.init_state_transition_probs(time_interval, sampling_num)

        # 報酬
        self.depths = self.depth_means(puddles, sampling_num)

        # その他
        self.time_interval = time_interval
        self.puddle_coef = puddle_coef

    def policy_evaluation_sweep(self):
        """方策評価による状態価値Vπの更新"""
        max_delta = 0.0
        for index in self.indexes: # 状態空間のインデックス
            if not self.final_state_flags[index]: # 終端でないなら
                # 方策に従った行動価値Qπ
                q = self.action_value(tuple(self.policy[index]), index)

                # 変化量
                delta = abs(self.value_function[index] - q)

                # 最大変化量
                max_delta = delta if delta > max_delta else max_delta

                # 行動価値Qπで状態価値Vπを更新
                self.value_function[index] = q

        return max_delta


    def action_value(self, action, index):
        """行動と状態インデックスから行動価値Q(s,a)を計算
        [1] indexに差分deltaを足して、はみだし処理後にタプルに変換
        [2] 報酬を計算
        [3] 行動価値 Q(s,a) = ∑_{s' in S} P(s'|s,a) { R(s,a,s') + Vπ(s')}
        """
        value = 0.0
        for delta, prob in self.state_transition_probs[(action, index[2])]: # index[2]: 方角のインデックス
            after = tuple(self.out_correction(np.array(index).T + delta)) # s'

            # この報酬計算でマイナス報酬は理解できるが、 - self.time_intervalをしている意味が理解できない.
            reward = - self.time_interval * self.depths[(after[0],after[1])] * self.puddle_coef - self.time_interval

            # 行動価値: Q(s,a) = ∑_{s' in S} P(s'|s,a) { R(s,a,s') + Vπ(s') }
            value += prob * (reward + self.value_function[after])

        return value

    def out_correction(self, index):
        """インデックスが状態遷移によって範囲をはみ出した時の処理
        位置 (x, y) : 現状の方策では範囲外になることはないので、無処理
        方向 (θ) : 0-360度に対応するインデックスに正規化してインデックスを返す.
        """
        index[2] = (index[2] + self.index_nums[2]) % self.index_nums[2] # 方向の処理
        return index

    def depth_means(self, puddles, sampling_num):
        """セル中の座標を均等にsampling_num**2点サンプリング"""
        dx = np.linspace(0, self.widths[0], sampling_num)
        dy = np.linspace(0, self.widths[1], sampling_num)
        samples = list(itertools.product(dx,dy))

        ### 深さの合計が計算されて格納される
        tmp = np.zeros(self.index_nums[0:2]) #深さの合計が計算されて入る
        for xy in itertools.product(range(self.index_nums[0]), range(self.index_nums[1])):
            for sx in samples:
                pose = self.pose_min + self.widths * np.array([xy[0],xy[1],0]).T + np.array([sx[0],sx[1],0]).T # セル内の座標
                for p in puddles:
                    tmp[xy] += p.depth * p.inside(pose) # 深さに水たまりの中か否か(1 or 0)をかけて足す

            tmp[xy] /= sampling_num**2 # 深さの合計から平均値に変換

        return tmp

    def init_state_transition_probs(self, time_interval, sampling_num):
        """セルの中の座標を均等にsampling_num**3点サンプリング"""
        # 隣のセルにはみ出さないように端を避ける
        dx = np.linspace(0.001, self.widths[0]*0.999, sampling_num) # 0.001 ~ Δx*0.999
        dy = np.linspace(0.001, self.widths[1]*0.999, sampling_num) # 0.001 ~ Δy*0.999
        dt = np.linspace(0.001, self.widths[2]*0.999, sampling_num) # 0.001 ~ 2pi*0.999
        samples = list(itertools.product(dx,dy,dt))

        ###各行動、各方角でサンプリングした点を移動してインデックスの増分を記録###
        tmp = {}
        for a in self.actions:
            for i_t in range(self.index_nums[2]):
                transitions = []
                for sx in samples:
                    before = np.array([
                        sx[0], # sampling_x 
                        sx[1], # sampling_y
                        sx[2] + i_t * self.widths[2] # sampling_theta
                    ]).T + self.pose_min # 遷移前の姿勢 (sx, sy, stheta)
                    
                    before_index = np.array([0, 0, i_t]).T # 遷移前のインデックス
                    after = IdealRobot.state_transition(nu=a[0], # 速度v
                                                        omega=a[1], # 角速度ω
                                                        time=time_interval, # 離散間隔Δt
                                                        pose=before, # 遷移前の姿勢(bx,by,bω)
                                                        ) # 遷移後の姿勢 (ax,ay,atheta)
                    
                    after_index = np.floor(
                        (after - self.pose_min) / self.widths
                    ).astype(int) # 遷移後のインデックス

                    transitions.append(after_index - before_index) # インデックスの差分

                unique, count = np.unique(transitions, axis=0, return_counts=True) # 集計(どのセルへの遷移が何回か)
                probs = [c.item()/sampling_num**3 for c in count] # サンプル数で割って確率にする
                tmp[a,i_t] = list(zip(unique, probs))

                """
                状態空間(state)と制御指令(action)はインデックスで管理する
                tmp = {
                    ((uv, uω), index_for_uω) = [
                        [delta_index_for_x_space, delta_index_for_y_space, delta_index_for_ω_space], 
                        [確率分布(0.01, 0.21, ..., )],
                    ] ,
                    ...,
                }
                """
        return tmp

    def init_policy(self):
        """方策(ロボットの行動)の初期化"""
        # 初期方策(生の値: 各状態における速度、角速度)
        init_value = np.zeros(np.r_[self.index_nums,2]) #制御出力が2次元なので、配列の次元を4次元に (px,py,theta,uv,uo) [L, N, M, 2]
        for index in self.indexes:
            center = self.pose_min + self.widths * (np.array(index).T + 0.5) # [0 1 ...30] + 0.5 = [0.5 1.5 ...30.5]
            """水たまりを無視して右上から左下のゴールに向かう方策(制御指令) """
            init_value[index] = PuddleIgnoreAgent.policy(center, self.goal) 
        
        return init_value

    def init_value_function(self):
        # 全離散状態を要素にもつ配列を作成
        v = np.empty(self.index_nums)
        f = np.zeros(self.index_nums)

        for index in self.indexes:
            f[index] = self.final_state(np.array(index).T)
            v[index] = self.goal.value if f[index] else -100.0

        return v, f

    def final_state(self, index):
        # 離散領域内の四隅の座標を計算
        x_min, y_min, _ = self.pose_min + self.widths * index # 左下
        x_max, y_max, _ = self.pose_min + self.widths * (index + 1) # 右上

        corners = [[x_min, y_min, _], [x_min, y_max, _], [x_max, y_min, _], [x_max, y_max, _]] # 四隅座標
        return all([self.goal.inside(np.array(c).T) for c in corners]) # 全部のgoal.insideがTrueであること



class DynamicProgramming:
    def __init__(self,
                 widths: np.ndarray,
                 goal: Goal,
                 puddles: List[Puddle],
                 time_interval: float, # [sec]
                 sampling_num: int,
                 puddle_coef: float = 100,
                 lowerleft: np.ndarray = np.array([-4,-4]).T,
                 upperright: np.ndarray = np.array([4,4]).T,
                 ):
        self.pose_min: np.ndarray = np.r_[lowerleft, 0.0] # 行方向の連結 (minx, miny, mintheta)
        self.pose_max: np.ndarray = np.r_[upperright, math.pi*2] # maxtheta: 2pi
        self.widths = widths # 離散化時の各状態の幅
        self.goal: Goal = goal

        # 状態空間の離散化(連続値をwidthsでインデックスにマッピング)
        self.index_nums: np.ndarray = ((self.pose_max - self.pose_min)/self.widths).astype(int)
        nx, ny, nt = self.index_nums # int, int, int
        self.indexes: List[Tuple[int,int,int]] = list(itertools.product(range(nx),range(ny),range(nt))) # 全離散状態のインデックスを計算

        # 状態価値 Vπ(s) = ∑_{a} π(a|s) ∑_{s} p(s'|s,a) { R(s',a,s) + Vπ(s') }
        self.value_function, self.final_state_flags = self.init_value_function()

        # 方策
        self.policy = self.init_policy() # [velocity,omega]の集合

        # 行動
        self.actions: List[Tuple[float,float]] = list(set([tuple(self.policy[i].tolist()) for i in self.indexes]))

        # 状態遷移確率
        self.state_transition_probs: Dict[Tuple[Tuple[float,float],int], List[List[int],List[float]]] \
            = self.init_state_transition_probs(time_interval, sampling_num)

        # 報酬
        self.depths: np.ndarray = self.depth_means(puddles, sampling_num)

        # その他
        self.time_interval: float = time_interval
        self.puddle_coef: float = puddle_coef

    def value_iteration_sweep(self) -> float:
        """価値反復による状態価値Vπと方策πの更新
        戦略：方策を考えることは、後回しにしてひたすら各状態でとりうるV(s)の最大値を求めていき、
        最後に収束したVから方策を選ぶ.
        """
        max_delta = 0.0
        for index in self.indexes:
            if not self.final_state_flags[index]:
                max_q = -1e100 # 最大行動価値
                max_a = None   # 最大行動価値に対応する行動

                # 状態sにおける全行動の行動価値を計算.
                qs = [self.action_value(a, index) for a in self.actions]

                # グリーディー化による最適行動価値
                max_q: float = max(qs)

                # グリーディー化による方策改善 max_a <- argmax Q(s,a)
                max_a: Tuple[float, float] = self.actions[np.argmax(qs)]

                # 変化量
                delta = abs(self.value_function[index] - max_q)

                # スイープ中で最大の変化量の更新
                max_delta = delta if delta > max_delta else max_delta

                # 最大行動価値で状態価値を更新
                self.value_function[index] = max_q
                # 最大行動価値に対応する行動で方策を更新
                self.policy[index] = np.array(max_a).T

        return max_delta
            

    def policy_evaluation_sweep(self):
        """方策評価による状態価値Vπ(s)の更新
        戦略： 方策π(a|s)からサンプリングした行動aから作った行動価値関数でVπ(s)を更新. 
        更新式：Vπ(s) <- Q(s,a~π(a|s)) 
        行動価値 Q(s,a) = ∑_{s' in S} P(s'|s,a) { R(s,a,s') + Vπ(s') }
        状態価値 Vπ(s) = ∑_{a in A} π(a|s) Q(s,a)
        """
        max_delta = 0.0

        # 全ての状態に対して状態価値を更新
        for index in self.indexes: # 状態空間のインデックス
            if not self.final_state_flags[index]: # 終端でないなら
                # 方策πに従った行動価値Qπ=状態価値Vπ
                q = self.action_value(tuple(self.policy[index]), index)

                # 状態価値の変化量
                delta = abs(self.value_function[index] - q)

                # 最大変化量
                max_delta = delta if delta > max_delta else max_delta

                self.value_function[index] = q

        return max_delta


    def action_value(self, action:Tuple[float,float], index: Tuple[int,int,int], out_penalty=True) -> float:
        """はみだしペナルティ + 行動と状態インデックスから行動価値Q(s,a)を計算
        [1] indexに差分deltaを足して、はみだし処理後にタプルに変換
        [2] 報酬を計算
        [3] 行動価値 Q(s,a) = ∑_{s' in S} P(s'|s,a) { R(s,a,s') + Vπ(s') }
        """
        value = 0.0 # 行動価値Q(s,a)
        for delta, prob in self.state_transition_probs[(action, index[2])]:
            # delta: [delta_index_for_x_space, delta_index_for_y_space, delta_index_for_ω_space]
            # probs: [標本確率分布]
            after_index, out_reward = self.out_correction(np.array(index).T + delta)
            after_index = tuple(after_index) # [index_x, index_y, index_theta]

            # マイナス報酬
            reward = - self.time_interval * self.depths[(after_index[0], after_index[1])] * self.puddle_coef - self.time_interval

            # 行動価値: Q(s,a) = ∑_{s' in S} P(s'|s,a) { R(s,a,s') + Vπ(s') }
            value += prob * (reward + self.value_function[after_index])

        return value

    def out_correction(self, index: np.ndarray) -> Tuple[np.ndarray, float]:
        out_reward = 0.0

        # 方角の処理
        index[2] = (index[2] + self.index_nums[2]) % self.index_nums[2]

        # 位置の処理
        for i in range(2): # 0,1
            if index[i] < 0:
                index[i] = 0 # 範囲内に丸め処理
                out_reward = -1e100 # 範囲外の報酬
            elif index[i] >= self.index_nums[i]:
                index[i] = self.index_nums[i] - 1 # 範囲内に丸め処理
                out_reward = -1e100 # 範囲外の報酬

        return index, out_reward

    def depth_means(self, puddles: List[Puddle], sampling_num: int) -> np.ndarray:
        """セル中の座標を均等にsampling_num**2点サンプリング"""
        dx = np.linspace(0, self.widths[0], sampling_num)
        dy = np.linspace(0, self.widths[1], sampling_num)
        samples: List[Tuple[float,float]] = list(itertools.product(dx,dy))

        # 深さの合計格納される
        tmp = np.zeros(self.index_nums[0:2])
        for xy in itertools.product(range(self.index_nums[0]), range(self.index_nums[1])): # (index_x, index_y)
            for sx in samples:
                pose = self.pose_min \
                    + self.widths * np.array([xy[0],xy[1],0]).T \
                    + np.array([sx[0],sx[1],0]).T # 各セル内のサンプリング点の座標

                for p in puddles:
                    tmp[xy] += p.depth * p.inside(pose) # 水たまりの中か否か(1or0) , 深さの合計

            tmp[xy] /= sampling_num**2 # 深さの合計から平均値に変換

        return tmp

    def init_state_transition_probs(self, time_interval: float, sampling_num: int):
        """セルの中の座標を均等にsampling_num**3点サンプリング"""
        dx = np.linspace(0.001, self.widths[0]*0.999, sampling_num)
        dy = np.linspace(0.001, self.widths[1]*0.999, sampling_num)
        dt = np.linspace(0.001, self.widths[2]*0.999, sampling_num)
        samples: List[Tuple[float,float,float]] = list(itertools.product(dx,dy,dt))

        # 各姿勢(x,y,theta)でサンプリングした点を移動してインデックスの増分を記録
        tmp = {}
        for a in self.actions: # 制御指令(uv,uω)
            for i_t in range(self.index_nums[2]): # 方角のインデックス
                transitions: List[np.array] = [] # List[Tuple[int,int,int]]
                for sx in samples:
                    # 方角のみ変化
                    before = np.array([
                        sx[0], # sampling_x
                        sx[1], # sampling_y
                        sx[2] + i_t * self.widths[2] # sampling_theta
                    ]).T + self.pose_min # 状態遷移前の姿勢

                    before_index = np.array([0,0,i_t]).T # 状態遷移前のインデックス
                    after = IdealRobot.state_transition(nu=a[0], # 制御指令速度v
                                                        omega=a[1], # 制御指令角速度ω
                                                        time=time_interval, # 離散時間間隔
                                                        pose=before,
                                                        ) # 状態遷移後の姿勢 (ax,ay,atheta)

                    after_index = np.floor((after-self.pose_min)/self.widths).astype(int)

                    transitions.append(after_index - before_index) # インデックスの差分

                # 集計(どのセルへの遷移が何回か)
                unique, count = np.unique(transitions, axis=0, return_counts=True)
                # サンプル数で割って確率にする
                probs = [c.item()/sampling_num**3 for c in count]

                # 状態遷移確率を格納
                tmp[a, i_t] = list(zip(unique, probs))
                """
                状態空間(state)と制御指令(action)はインデックスで管理する
                tmp = {
                    ((uv, uω), index_for_uω) = [
                        [delta_index_for_x_space, delta_index_for_y_space, delta_index_for_ω_space], 
                        [標本確率分布(0.01, 0.21, ..., )],
                    ] ,
                    ...,
                }
                """
        return tmp

    def init_policy(self) -> np.ndarray:
        """方策の初期化"""
        # 初期方策 = (速度指令, 角速度指令)@各状態
        # 制御出力が2次元なので、離散状態空間(3次元)に1次元(uv,uω)を加える. [NX,NY,NT,NU] = [nx,ny,nt,2]
        init_value = np.zeros(np.r_[self.index_nums, 2])
        for index in self.indexes:
            center = self.pose_min + self.widths * (np.array(index).T + 0.5)

            # ロボット方策による初期化
            init_value[index] = PuddleIgnoreAgent.policy(center, self.goal)

        return init_value

    def init_value_function(self) -> Tuple[np.ndarray, np.ndarray]:
        """全離散状態sにおける状態価値Vπの初期値"""
        v = np.empty(self.index_nums) # 状態価値
        f = np.zeros(self.index_nums) # 終端フラグ

        for index in self.indexes: # List[Tuple[int,int,int]]
            f[index] = self.final_state(np.array(index).T)
            v[index] = self.goal.value if f[index] else -100.0

        return v, f

    def final_state(self, index: np.ndarray) -> bool:
        """離散状態領域内の四隅の座標を計算"""

        # 離散領域(3DBox)の左下と右上の連続値としての座標を計算
        x_min, y_min, _ = self.pose_min + self.widths * index # 左下
        x_max, y_max, _ = self.pose_min + self.widths * (index + 1) # 右上

        corners = [
            [x_min, y_min, _], [x_min, y_max, _], [x_max, y_min, _], [x_max, y_max, _]
        ] # 四隅座標

        return all([self.goal.inside(np.array(c).T) for c in corners]) # 終端状態か否か