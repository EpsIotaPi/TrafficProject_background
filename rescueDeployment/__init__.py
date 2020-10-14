

import numpy as np

# 节点类
class Node:
    def __init__(self, id, name, long, lati):
        self.id = id
        self.name = name
        self.longitude = long  # 经纬度
        self.latitude = lati


# 路径类
class Path:
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
        self.route_index = arr  # 路径节点序号

        self.carNum = carNum
        self.route_node = []
        for node_index in self.route_index:
            self.route_node.append(Node_list[node_index])  # 路径节点列表


# 方案类
class Plan:
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



