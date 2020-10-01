import pulp
# import numpy as np
# from pprint import pprint
import numpy as np
# from scipy import spatial
import pandas as pd
# import matplotlib.pyplot as plt

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
    prob.solve()
    return {'objective':pulp.value(prob.objective), 'var': [[pulp.value(var[i][j]) for j in range(col)] for i in range(row)]}

def get_input(accident_index):
    """
    Calculate the vehicles to be dispatched to the accident point

    :param accident_index: np.array([1, 2, 3])
    :return: a martix will indicate from Rescue point to accident point
    """
    accident_index=accident_index-1

    sheet1 = pd.read_excel('./rescueDeployment/result.xls', sheet_name='sheet1', header=None)
    goal_value_matrix = sheet1.values

    #返回路径？
    sheet2 = pd.read_excel('./rescueDeployment/result.xls', sheet_name='sheet2', header=None)
    route_matrix = sheet2.values

    arr= np.zeros(6)   #救援点的个数固定为6
    goal_funtion_matrix=goal_value_matrix[:,accident_index]
    route_matrix=route_matrix[:,accident_index]
    costs=np.column_stack((np.array(1/np.array(goal_funtion_matrix)),arr))

    max_plant = [2, 3, 1, 2, 1, 2]  # 救援点对应存储的车辆数
    max_cultivation = []  # 事故点对应的需求数
    # 事故点的需求车辆默认为从2开始，步长为1，递增
    count=2
    for i in range(len(accident_index)):  #事故点不能超过3
        max_cultivation.append(count)
        count=count+1
    max_cultivation.append(sum(max_plant)-sum(max_cultivation))
    res = transportation_problem(costs, max_plant, max_cultivation)
    return res['var'],route_matrix
    # print(f'最大值为{res["objective"]}')
    # print('各变量的取值为：')
    # pprint(res['var'])




if __name__ == '__main__':
    #result.xls中，sheet1存储目标函数值，sheet2存储路径，sheet3存储对应的行程距离(km)，sheet4存储对应的时间(h)
    #救援点的位置为[19,29,34,15,45,33]
    #救援点对应的存储车辆数目为[2, 3, 1, 2, 1, 2]
    # 事故点不能超过3个,事故点的需求车辆默认为从2开始，步长为1，递增
    accident_index = np.array([21, 10, 17])#事故点的位置
    a, b = get_input(accident_index)
    print(np.array(a))
    print(b)

