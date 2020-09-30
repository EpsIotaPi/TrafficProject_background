# import numpy as np
from numpy import *
# from scipy import spatial
import pandas as pd
# import matplotlib.pyplot as plt


#数据读取和预处理
sheet1 = pd.read_excel('路网数据改进.xlsx',sheet_name= '距离矩阵',header=None)
distance_matrix = sheet1.values
distance_matrix=delete(distance_matrix,[0,1],axis=0)
distance_matrix=delete(distance_matrix,[0,1],axis=1)

sheet2 = pd.read_excel('路网数据改进.xlsx',sheet_name= '通行能力',header=None)
traffic_capacity_matrix = sheet2.values
traffic_capacity_matrix=delete(traffic_capacity_matrix,[0,1],axis=0)
traffic_capacity_matrix=delete(traffic_capacity_matrix,[0,1],axis=1)

sheet3 = pd.read_excel('路网数据改进.xlsx',sheet_name= '实际交通流量',header=None)
Actual_traffic_flow_matrix = sheet3.values
Actual_traffic_flow_matrix=delete(Actual_traffic_flow_matrix,[0,1],axis=0)
Actual_traffic_flow_matrix=delete(Actual_traffic_flow_matrix,[0,1],axis=1)

sheet4 = pd.read_excel('路网数据改进.xlsx',sheet_name= '发生事故的概率',header=None)
accident_prob_matrix = sheet4.values
accident_prob_matrix=delete(accident_prob_matrix,[0,1],axis=0)
accident_prob_matrix=delete(accident_prob_matrix,[0,1],axis=1)

sheet5 = pd.read_excel('路网数据改进.xlsx',sheet_name= '平面坐标',header=None)
position_matrix = sheet5.values
position_matrix=delete(position_matrix,[0,1],axis=1)

# print(distance_matrix[0, :])
# a=distance_matrix[0, :]
# print(np.nan_to_num(a))
def nan_to_zero(matrix_a):
    m=matrix_a.shape[0]
    for i in range(m) :
        for j in range(m):
            if(matrix_a[i,j]!=matrix_a[i,j]):
                matrix_a[i,j]=0
    return matrix_a

def matrix_plus(matrix_a):
    a = np.zeros((matrix_a.shape[0],matrix_a.shape[0]))
    for i in range(matrix_a.shape[0]):
        a[:,i]=matrix_a[i,:]
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_a.shape[0]):
            matrix_a[i, j]=matrix_a[i,j]+a[i,j]
    return matrix_a

distance_matrix=nan_to_zero(distance_matrix)
traffic_capacity_matrix=nan_to_zero(traffic_capacity_matrix)
Actual_traffic_flow_matrix=nan_to_zero(Actual_traffic_flow_matrix)
accident_prob_matrix=nan_to_zero(accident_prob_matrix)

traffic_capacity_matrix=matrix_plus(traffic_capacity_matrix)
Actual_traffic_flow_matrix=matrix_plus(Actual_traffic_flow_matrix)
accident_prob_matrix=matrix_plus(accident_prob_matrix)
#------------数据读取和预处理完成--------------

# 参数设置
num_points=45  #城市数量
# start_point=None   #起点
# end_point=None   #终点
k=10 #蚂蚁数量
maxN=100 #迭代次数
#BPR=t0*(1+alpha*(Q/C)**belta) 自由流速度假定为80km/h  x1=alpha,x2=belta
x1=0.5668
x2=1.4431
BPR=distance_matrix/80*(1+x1*(Actual_traffic_flow_matrix/(traffic_capacity_matrix+1e-10))**x2)
# 初始信息素浓度 tau=埃普西隆+埃普西隆0 ，x3=埃普西隆，x4=埃普西隆0
x3=1
x4=2
# 概率转移公式中，Tau**alpha * yita**beta * weight**gama  x5=alpha,x6=beta,x7=gama
x5=0.3
x6=1
x7=0.6
#启发函数中 yita = 1 / (（u1 * dij + u2 * dja +u3 * delta theta / pi） * eij)
#  x8=u1, x9=u2,x10=u3
x8=1
x9=0.4
x10=2
# 概率转移中的分类选择参数 q1,q2 ,x11=q1,x12=q2
x11=0.1
x12=0.9
# 目标函数Z = w1 * BPR + w2 * accident_prob_matrix,x13=w1,x14=w2
x13=0.9
x14=5
# 信息素挥发系数 rho=k*Nmax/(Nmax+n) ,tau=(1-rho)t+u*log(1+delta_tau) x15=k,x16=u
x15=0.3
x16=1.2
#传统蚁群算法的信息素挥发系数
x17=0.5

#目标函数 Z = w1 * BPR + w2 * accident_prob_matrix,
def cal_total_goal(routine, num_points):
    if num_points == 0:
        return inf
    sum = 0
    for i in range(num_points):
        x = x13 * BPR[routine[i], routine[i + 1]] + x14 * accident_prob_matrix[routine[i], routine[i + 1]]
        sum = sum + x
    return sum

# ——————————————调用——————————————
from ACA_update import ACA_TSP
rescue_list=[19,29,34,15,45,33]  #rescur index

# accident_list=[21,10,17]   #accident index
#accident_list = filter(lambda x: (x not in rescue_list), a=np.arange(1,46,1))  #accident index
accident_list=np.arange(1,46,1)
goal_funtion_matrix=np.zeros((len(rescue_list),len(accident_list))) #存储目标函数值

total_route_list=[]
i_pointer=0
for i in rescue_list:
    j_pointer = 0
    route_list=[]
    for j in accident_list:
        if j not in rescue_list:
            start_point=i-1
            end_point=j-1
            # 初始化信息素浓度
            Tau = np.zeros((num_points, num_points))
            neighbor_index = np.nonzero(distance_matrix[:, end_point])[0]
            for k in range(num_points):
                Tau[k] = np.array([x3] * num_points)
                if k in neighbor_index:
                    # Tau[k,:] += x4
                    # Tau[:,k] += x4
                    for h in range(num_points):
                        if h in neighbor_index:
                            Tau[k, h] += x4 / 2
                            Tau[h, k] += x4 / 2
                        else:
                            Tau[k, h] += x4
                            Tau[h, k] += x4
            aca = ACA_TSP(func=cal_total_goal, n_dim=num_points,
                          start=start_point, end=end_point,
                          size_pop=k, max_iter=maxN,
                          distance_matrix=distance_matrix, BPR=BPR,
                          accident_prob_matrix=accident_prob_matrix,
                          position_matrix=position_matrix,
                          Tau=Tau,
                          alpha=x5, beta=x6, gama=x7,
                          u1=x8, u2=x9, u3=x10,
                          q1=x11, q2=x12,
                          k=x15, u4=x16
                          )
            best_x, best_y ,y_best_history,valid_node= aca.run()
            goal_funtion_matrix[i_pointer][j_pointer]=best_y
            route_list.append(str(best_x[:valid_node] + 1))
            print(best_x[:valid_node] + 1)  # 路线
            print(best_y)  # 目标函数值
        else:
            goal_funtion_matrix[i_pointer][j_pointer] = 0.
            route_list.append(str([]))
       # print(route_list)
        j_pointer = j_pointer + 1
    total_route_list.append(route_list)
    i_pointer=i_pointer+1

import xlwt  # 负责写excel
import xlrd
filename =xlwt.Workbook() #创建工作簿
sheet1 = filename.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
[h,l]=goal_funtion_matrix.shape #h为行数，l为列数
for i in range (h):
    for j in range (l):
        sheet1.write(i,j,goal_funtion_matrix[i,j])
filename.save('result.xls')

sheet2 = filename.add_sheet(u'sheet2',cell_overwrite_ok=True) #创建sheet
for i in range (h):
    for j in range (l):
        sheet2.write(i,j,total_route_list[i][j])
filename.save('result.xls')


#尝试
total_distance_matrix=np.zeros((h,l)) #存储路径行程，单位km
for i in range(h):
    for j in range(l):
        sum=0
        str=total_route_list[i][j]
        str = str.strip('[')
        str = str.strip(']')
        list = [int(s) for s in str.split() if s.isdigit()]
        arr=np.array(list)
        arr=arr-1
        for k in range(len(list)-1):
            sum=sum+distance_matrix[arr[k],arr[k+1]]
        total_distance_matrix[i,j]=sum
print(total_distance_matrix)
sheet3 = filename.add_sheet(u'sheet3',cell_overwrite_ok=True) #创建sheet
for i in range (h):
    for j in range (l):
        sheet3.write(i,j,total_distance_matrix[i,j])
filename.save('result.xls')

total_time_matrix=np.zeros((h,l)) #存储路径时间，单位h
for i in range(h):
    for j in range(l):
        sum=0
        str=total_route_list[i][j]
        str = str.strip('[')
        str = str.strip(']')
        list = [int(s) for s in str.split() if s.isdigit()]
        arr=np.array(list)
        arr=arr-1
        for k in range(len(list)-1):
            sum=sum+BPR[arr[k],arr[k+1]]
        total_time_matrix[i,j]=sum
print(total_time_matrix)
sheet4 = filename.add_sheet(u'sheet4',cell_overwrite_ok=True) #创建sheet
for i in range (h):
    for j in range (l):
        sheet4.write(i,j,total_time_matrix[i,j])
filename.save('result.xls')
