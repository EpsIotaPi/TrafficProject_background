# 蚁群算法和节点、路径、方案类的定义

from numpy import *
# import matplotlib.pyplot as plt
import numpy as np
import math
import json


# 蚁群算法
class ACA_TSP:
    def __init__(self,func, n_dim,start,end,
                 size_pop=10, max_iter=20,
                 distance_matrix=None,
                 BPR=None,
                 accident_prob_matrix=None,
                 position_matrix=None,
                 Tau=None,
                 alpha=1, beta=2, gama=1, #Tau**alpha * yita**beta * weight**gama
                 u1=1,u2=1,u3=1,      #启发函数 yita = 1 / (（u1 * dij + u2 * dja +u3 * delta theta / pi） * eij)
                 q1=0.3,q2=0.5,   #概率转移中的分类选择参数
                 k=1,u4=1  # rho=k*Nmax/(Nmax+n)  ,tau=(1-rho)t+u*log(1+delta_tau)
                 ):
        self.func = func
        self.n_dim = n_dim  # 城市数量
        self.size_pop = size_pop  # 蚂蚁数量
        self.max_iter = max_iter  # 迭代次数
        self.alpha = alpha  # 信息素重要程度  转移概率公式中用到
        self.beta = beta  # 适应度的重要程度  转移概率公式中用到
        self.gama=gama # 权重的重要程度  转移概率公式中用到
        # self.rho = rho  # 信息素挥发速度  信息素更新公式用到
        self.start=start
        self.end=end
        self.distance_matrix=distance_matrix
        self.BPR = BPR
        self.accident_prob_matrix=accident_prob_matrix
        self.position_matrix=position_matrix
        #权重
        self.weight=1/(BPR+1e-10 * np.ones((n_dim, n_dim)))
        # 启发函数的参数 yita = 1 / (（u1 * dij + u2 * dja +u3 * delta theta / pi） * eij)
        self.u1=u1
        self.u2=u2
        self.u3=u3
        #概率转移中的分类选择参数
        self.q1=q1
        self.q2=q2
        # 信息素挥发系数 rho=k*Nmax/(Nmax+n)
        # 信息素更新公式 tau=(1-rho)t + u*log(1+delta_tau)
        self.k=k
        self.u4=u4
        # 信息素矩阵 n_dim * n_dim
        self.Tau=Tau
        # 路径矩阵，蚂蚁数量*城市数量 大小的矩阵，第i行是第i只蚂蚁的路径id
        self.Table = np.zeros((size_pop, n_dim)).astype(np.int)  # 某一代每个蚂蚁的爬行路径
        self.y = None  # 某一代每个蚂蚁的爬行的目标函数值
        self.x_best_history, self.y_best_history = [], []  # 记录当前迭代为止取得的最佳情况
        self.x_best, self.y_best = None, inf # 记录当前迭代的最优结果
        self.Table_len = [0] * self.size_pop # 因为路径长度不确定，要记录每只蚂蚁路径的长度
        self.best_len=None
        self.best_len2=None

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        times=0
        for i in range(self.max_iter):  # 对每次迭代
            for j in range(self.size_pop):  # 对每个蚂蚁
                self.Table[j, 0] = self.start  # 开始结点
                self.Table_len[j] = 0
                for k in range(self.n_dim - 1):  # 蚂蚁到达的每个节点
                    taboo_set = set(self.Table[j, :k + 1])  # 已经经过的点和当前点，不能再次经过  把前k个结点加入到禁忌表
                    allow_list = list(set(range(self.n_dim)) - taboo_set)  # 在这些点中做选择   不在禁忌表中的结点可以去
                    allow_index = np.nonzero(self.distance_matrix[self.Table[j, k], :])[0] #与原实现有差别，没有路的不选
                    allow_list=list(set(allow_list).intersection(set(allow_index)))
                    # 计算启发函数值
                    yita = []
                    if (len(allow_list) == 0):
                        self.Table_len[j] = 0
                        self.Table[j] = 0
                        break
                    else:
                        for h in allow_list:
                            # yita = 1 / (（u1 * dij + u2 * dja +u3 * delta theta / pi） * eij)
                            a1=self.u1*self.distance_matrix[self.Table[j, k],h]
                            a2=self.u2*np.sqrt((self.position_matrix[h,0]-self.position_matrix[self.end,0])**2+
                                                      (self.position_matrix[h,1]-self.position_matrix[self.end,1])**2)
                            if(h!=self.end):
                                # print(self.position_matrix[self.end, 0])
                                # print(self.position_matrix[h, 0])
                                a3= self.u3/math.pi*np.abs(math.atan((self.position_matrix[h,1]-self.position_matrix[self.Table[j, k],1])/
                                                                        (self.position_matrix[h,0]-self.position_matrix[self.Table[j, k],0]))-
                                                               math.atan((self.position_matrix[self.end, 1] - self.position_matrix[h, 1]) /
                                                                      (self.position_matrix[self.end, 0] - self.position_matrix[h, 0])))
                            else:
                                a3 = 0
                            yita.append(1/((a1+a2+a3) *self.accident_prob_matrix[self.Table[j, k],h]))

                    # 计算概率转移公式 Tau**alpha * yita**beta * weight**gama
                    List1=list(map(lambda x: x ** self.alpha, self.Tau[self.Table[j, k], allow_list]))
                    List2=list(map(lambda x: x ** self.beta, yita))
                    List3=list(map(lambda x: x ** self.gama, self.weight[self.Table[j, k], allow_list]))
                    prob= np.multiply(np.multiply(List1, List2), List3)
                    # max{tau**alpha * yita**belta * w**gama}
                    max_hybrid_index=np.argmax(prob)
                    # max{w**gama}
                    max_weight_index=np.argmax(List3)
                    prob = prob / prob.sum()  # 概率归一化
                    r=random.random()
                    # 选择max{tau**alpha * yita**belta * w**gama}
                    if r<self.q1:
                        # print('r1(%d)'%(r))
                        next_point=allow_list[max_hybrid_index]
                    elif self.q1<r<self.q2:
                        # 选择下一个城市节点 轮盘赌法
                        next_point = np.random.choice(allow_list, size=1, p=prob)[0]  # 从给定的一维数组中生成一个随机样本，根据prob概率
                    else:
                        # 选择max{w**gama}
                        next_point=allow_list[max_weight_index]
                    self.Table[j, k + 1] = next_point
                    self.Table_len[j] = self.Table_len[j] +1
                    if(next_point==self.end):# 到达终点
                        self.Table[j,k+2:]=0
                        break

            # 目标函数值
            y = np.array([self.func(p,len) for p,len in zip(self.Table,self.Table_len)])

            flag=0
            # best_history存储当前迭代得到的最优解
            if(y.min() != inf):  # 这一轮迭代都是无效路径，不存储最好路径，不更新信息素
                flag=1
                index_best = y.argmin()  # y路径最短
                self.best_len=self.Table_len[index_best]
                sum=0
                for z in y:
                    if(z!=inf):
                        sum+=z
                Z_avg=sum/y.size
                self.x_best, self.y_best = self.Table[index_best, :].copy(), y[index_best].copy() # 存储最好情况的路径和距离

                # 激励度phi=1+(Zavg-Zk)/(Zavg-Zb)
                phi=[]
                for z in y:
                    if z<Z_avg:
                        phi.append( 1 + (Z_avg - z) / (Z_avg - self.y_best+0.1))  #分母加0.1，防止除0错误
                    else:
                        phi.append(1)
                # 计算需要新涂抹的信息素
                delta_tau = np.zeros((self.n_dim, self.n_dim))
                for j in range(self.size_pop):  # 每个蚂蚁
                    for k in range(self.Table_len[j]):  # 每个节点
                        n1, n2 = self.Table[j, k], self.Table[j, k + 1]  # 蚂蚁从n1节点爬到n2节点
                        delta_tau[n1, n2] += math.log2(1+1*phi[j] / y[j])  # 涂抹的信息素  1*phi/爬行距离

                # 信息素飘散+信息素涂抹  信息素更新公式
                # rho=k*Nmax/(Nmax+n)
                # tau=(1-rho)t+u*log(1+delta_tau)
                self.Tau = (1 - self.k*self.max_iter/(self.max_iter+i)) * self.Tau + self.u4*(1+delta_tau)
            if times==0:
                if flag==1:
                    self.x_best_history.append(self.x_best)
                    self.y_best_history.append(self.y_best)
                    self.best_len2=self.best_len
                else:
                    self.x_best_history.append([])
                    self.y_best_history.append(inf)
            elif self.y_best<self.y_best_history[len(self.y_best_history)-1]:
                self.x_best_history.append(self.x_best)
                self.y_best_history.append(self.y_best)
                self.best_len2=self.best_len
            else:
                self.x_best_history.append(self.x_best_history[len(self.y_best_history)-1])
                self.y_best_history.append(self.y_best_history[len(self.y_best_history)-1])
            times+=1

        # y_best_history只存储当前得到的最好结果
        if len(self.y_best_history):
            best_x = self.x_best_history[len(self.y_best_history)-1]
            best_y = self.y_best_history[len(self.y_best_history)-1]
            return best_x, best_y, self.y_best_history,self.best_len2+1
        else:
            return [],inf,[],self.best_len2



