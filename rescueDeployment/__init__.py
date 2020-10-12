from .ACA_update import *
from .transportation import *

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

if __name__ == '__main__':
    #result.xls中，sheet1存储目标函数值，sheet2存储路径，sheet3存储对应的行程距离(km)，sheet4存储对应的时间(h)
    #救援点的位置为[19,29,34,15,45,33]
    #救援点对应的存储车辆数目为[2, 3, 1, 2, 1, 2]
    # 事故点不能超过3个,事故点的需求车辆默认为从2开始，步长为1，递增
    # accident_index=np.array([21,10,17])#事故点的位置

    accident_point_list = [21,10,17]
    result_plan = rescuePlan_API(accident_point_list)
    print("-----调试看变量-----")
    print(result_plan) # 运行5s，调试看变量
    print(result_plan[0].path_list[0])

    # TODO:要排除生成的其他方案和最佳方案一样的情况
