

import numpy as np
from DBmanage import accessDatabase


class coordinate:
    def __init__(self, longitude:float, latitude:float):
        self.longitude = longitude
        self.latitude = latitude


# 节点类
class Node:
    def __init__(self, id):
        sqlStatement = "select * from Points where p_id = " + id
        gripdata = accessDatabase(sqlStatement)
        for i in gripdata:
            self.id = int(gripdata[0])
            self.name = gripdata[1]
            self.coordinate = coordinate(float(gripdata[2]), float(gripdata[3]))
            self.traffic_rate = float(gripdata[5])


# 路径类
class Path:
    serial_number = ''

    def __init__(self, time: float, distance: float, route: str, carNum: int, Node_list: [Node]):
        """

        :param time: hour
        :param distance: km
        :param route: Like:'[29, 28, 5, 4, 21]'
        :param carNum:
        :param Node_list:
        """
        self.time = time  # 行驶时间
        self.distance = distance  # 行驶路程

        str = route
        str = str.strip('[')
        str = str.strip(']')
        list = [int(s) for s in str.split() if s.isdigit()]
        arr = np.array(list)
        arr = arr - 1
        route_index = arr  # 路径节点序号

        self.carNum = carNum
        self.route_node = []
        for node_index in route_index:
            self.route_node.append(Node_list[node_index])  # 路径节点列表

        self.start_id = route_index[0]
        self.end_id = route_index[len(route_index) - 1]

    def calCongestionRate(self) -> int:
        """
        返回平均拥堵率
        :return:
        """
        rate = 0
        for node in self.route_node:
            rate += node.traffic_rate
        return int(rate / len(self.route_node))



# 方案类
class Plan:
    compare_avgTime = 0
    compare_avgDis = 0
    is_fast = False
    is_short = False

    def __init__(self, path_list: [Path]):
        self.path_list = path_list
        max_time = 0
        sum_time = 0
        sum_distance = 0
        for path in path_list:
            sum_time = sum_time + path.time
            sum_distance = sum_distance + path.distance
            if path.time > max_time:
                max_time = path.time
        self.max_time = max_time
        self.sum_time = sum_time  # 每条路径时间和
        self.avg_time = sum_time / len(path_list)
        self.sum_distance = sum_distance
        # 运输问题的目标函数值



class allPlans:
    def __init__(self, Plans: [Plan]):
        self.plan_list = Plans
        self.count = len(Plans)

        time_count = distance_count = 0
        for i in Plans:
            time_count += i.sum_time
            distance_count += i.sum_distance
        self.avgTime = time_count / self.count
        self.avgDis = distance_count / self.count

    def self_compare(self):
        fastest = shortest = 0
        for i in range(0, self.count):
            if self.plan_list[i].sum_time < self.plan_list[fastest].sum_time:
                fastest = i
            if self.plan_list[i].sum_distance < self.plan_list[shortest].sum_distance:
                shortest = i
            self.plan_list[i].compare_avgTime = int((self.plan_list[i].sum_time - self.avgTime) / self.avgTime)
            self.plan_list[i].compare_avgDis = int((self.plan_list[i].sum_distance - self.avgDis) / self.avgDis)

    def giveSerial(self, pos_id:[int], serial:[str]):
        for plan in self.plan_list:
            for path in plan.path_list:
                index = 0
                while(pos_id[index] != path.end_id):
                    index += 1
                path.serial_number = serial[index]














