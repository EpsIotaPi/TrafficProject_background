# 求解派遣方案，封装返回接口数据
# import sys
# sys.path.append('~/JetBrains/PycharmProjects/TrafficProject')

import pulp
from numpy import *
import pandas as pd
from rescueDeployment import *



# 求解运输问题的规划算法，用于确定派遣方案
def transportation_problem(costs, x_max, y_max):
    row = len(costs)
    col = len(costs[0])
    prob = pulp.LpProblem('Transportation Problem', sense=pulp.LpMaximize)
    var = [[pulp.LpVariable(f'x{i}{j}', lowBound=0, cat=pulp.LpInteger) for j in range(col)] for i in range(row)]
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    prob += pulp.lpDot(flatten(var), costs.flatten())
    for i in range(row):
        prob += (pulp.lpSum(var[i]) <= x_max[i])
    for j in range(col):
        prob += (pulp.lpSum([var[i][j] for i in range(row)]) <= y_max[j])
    # prob.writeLP("WhiskasModel.lp")
    prob.solve()
    return {'objective':pulp.value(prob.objective), 'var': [[pulp.value(var[i][j]) for j in range(col)] for i in range(row)]}

# 派遣方案矩阵封装到路径类
def description_scheme(res,route_matrix,distance_matrix,time_matrix):
    # 读取节点数据
    sheet5 = pd.read_excel('./rescueDeployment/sheets/路网数据改进.xlsx', sheet_name='平面坐标', header=None)
    node_data = sheet5.values
    node_data = delete(node_data, [0], axis=1)
    node_list =[]
    for i in range(45):
        node = Node(i, node_data[i,0], node_data[i,1], node_data[i,2])
        node_list.append(node)
    res = np.array(res) # 派遣方案矩阵
    accident_num = res.shape[1]-1
    rescue_num = res.shape[0]
    result_path = []
    # 对于每一列 每个事故点
    for i in range(accident_num):
        for j in range(rescue_num):
            if(res[j,i] != 0):
                path = Path(time_matrix[j,i],distance_matrix[j,i],route_matrix[j,i],res[j,i],node_list)
                result_path.append(path)
    return result_path

# 根据事故点确定派遣方案
def get_input(accident_index,result_file):
    accident_index=accident_index-1
    # 权值矩阵 路线 时间 距离，救援点个数 储备数，全部已知或固定
    sheet1 = pd.read_excel(result_file, sheet_name='sheet1', header=None)# 权值矩阵
    goal_value_matrix = sheet1.values

    #返回路径？
    sheet2 = pd.read_excel(result_file, sheet_name='sheet2', header=None)# 路线
    route_matrix = sheet2.values
    sheet3 = pd.read_excel(result_file, sheet_name='sheet3', header=None)# 距离
    distance_matrix = sheet3.values
    sheet4 = pd.read_excel(result_file, sheet_name='sheet4', header=None)# 时间
    time_matrix = sheet4.values

    arr= np.zeros(6)   #救援点的个数固定为6
    goal_funtion_matrix=goal_value_matrix[:,accident_index]
    route_matrix=route_matrix[:,accident_index]
    distance_matrix=distance_matrix[:,accident_index]
    time_matrix=time_matrix[:,accident_index]


    costs=np.column_stack((np.array(1/np.array(goal_funtion_matrix)),arr))

    max_plant = [2, 3, 1, 2, 1, 2]  # 救援点对应存储的车辆数

    max_cultivation = []  # 事故点对应的需求数
    # 事故点的需求车辆默认为从2开始，步长为1，递增
    count=2
    for i in range(len(accident_index)):  #事故点不能超过3
        max_cultivation.append(count)
        count=count+1
    max_cultivation.append(sum(max_plant)-sum(max_cultivation))
    # 构建costs矩阵，和各需求 供应数，转化为运输问题
    res = transportation_problem(costs, max_plant, max_cultivation)
    # 描述派遣方案

    # 计算运输问题目标函数值
    def cal_Obj(costs, res):
        result = costs * res
        Obj_value = result.sum()
        return Obj_value

    Obj_value = cal_Obj(costs, res['var'])
    # print("-----------------")
    # print(res['var'])
    # print("best_Obj_value", Obj_value)

    return res['var'],route_matrix,distance_matrix,time_matrix,costs
    # print(f'最大值为{res["objective"]}')
    # print('各变量的取值为：')
    # pprint(res['var'])

# 生成其他派遣方案
def get_input_2(accident_index,costs_real):
    accident_index=accident_index-1
    # 权值矩阵 路线 时间 距离，救援点个数 储备数，全部已知或固定
    sheet1 = pd.read_excel('./rescueDeployment/sheets/result.xls', sheet_name='sheet1', header=None)# 权值矩阵
    goal_value_matrix = sheet1.values

    #返回路径？
    sheet2 = pd.read_excel('./rescueDeployment/sheets/result.xls', sheet_name='sheet2', header=None)# 路线
    route_matrix = sheet2.values
    sheet3 = pd.read_excel('./rescueDeployment/sheets/result.xls', sheet_name='sheet3', header=None)# 距离
    distance_matrix = sheet3.values
    sheet4 = pd.read_excel('./rescueDeployment/sheets/result.xls', sheet_name='sheet4', header=None)# 时间
    time_matrix = sheet4.values

    arr= np.zeros(6)   #救援点的个数固定为6
    goal_funtion_matrix=goal_value_matrix[:,accident_index]
    route_matrix=route_matrix[:,accident_index]
    distance_matrix=distance_matrix[:,accident_index]
    time_matrix=time_matrix[:,accident_index]

    cost_ori = np.array(1/np.array(goal_funtion_matrix))

    max_plant = [2, 3, 1, 2, 1, 2]  # 救援点对应存储的车辆数

    max_cultivation = []  # 事故点对应的需求数
    # 事故点的需求车辆默认为从2开始，步长为1，递增
    count=2
    for i in range(len(accident_index)):  #事故点不能超过3
        max_cultivation.append(count)
        count=count+1
    max_cultivation.append(sum(max_plant)-sum(max_cultivation))

    # 计算运输问题目标函数值
    def cal_Obj(costs,res):
        result = costs*res
        Obj_value = result.sum()
        return Obj_value

    # 生成其他派遣方案
    res_list =[]
    Obj_value_list = []
    for i in range(100): # 把costs矩阵打乱成假的costs矩阵，再用运输问题优化函数，得到假的costs矩阵下的最优派遣方案
        cost_ori = cost_ori.T
        for j in range(3): # 打乱costs矩阵
            np.random.shuffle(cost_ori)
            cost_ori = cost_ori.T
        costs = np.column_stack((cost_ori, arr))
        res = transportation_problem(costs, max_plant, max_cultivation)
        Obj_value = cal_Obj(costs_real,res['var']) # 用真实costs矩阵，评估新生成派遣方案的目标函数值
        res_list.append(res['var'])
        Obj_value_list.append(Obj_value)
    # 选 运输问题目标函数值 最好的两个
    index_best = 0
    index_best_last = 0
    for i in range(100):
        if Obj_value_list[i] > Obj_value_list[index_best]:
            index_best_last = index_best
            index_best = i
    res_best = []
    Obj_value_best = []
    res_best.append(res_list[index_best])
    res_best.append(res_list[index_best_last])
    Obj_value_best.append(Obj_value_list[index_best])
    Obj_value_best.append(Obj_value_list[index_best_last])
    # print(res_best)
    # print(Obj_value_best)

    return res_best,route_matrix,distance_matrix,time_matrix

def rescuePlan_API(accident_point_list):
    accident_index = np.array(accident_point_list)  # 事故点的位置

    result_plan = []

    # 运输问题目标下 最优派遣方案
    res, route_matrix, distance_matrix, time_matrix, costs_real = get_input(accident_index, "./rescueDeployment/sheets/result.xls")
    path_list = description_scheme(np.array(res), route_matrix, distance_matrix, time_matrix)
    plan = Plan(path_list)
    result_plan.append(plan)
    best_sum_time = plan.sum_time

    # 运输问题目标下 生成的其他派遣方案中最优的两个
    res_list, route_matrix, distance_matrix, time_matrix = get_input_2(accident_index, costs_real)
    for res in res_list:
        path_list = description_scheme(np.array(res), route_matrix, distance_matrix, time_matrix)
        plan = Plan(path_list)
        if(plan.sum_time != best_sum_time):
            result_plan.append(plan)

    # 单对单次优路径下，最优派遣方案
    # res, route_matrix, distance_matrix, time_matrix, costs_real = get_input(accident_index, "result1.xls")
    # path_list = description_scheme(np.array(res), route_matrix, distance_matrix, time_matrix)
    # plan = Plan(path_list)
    # result_plan.append(plan)
    return result_plan  # 运行5s，调试看变量