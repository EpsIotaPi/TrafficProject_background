
import numpy as np
from dataBaseManage import accessDatabase, coordinate, Point
from rescueDeployment.transportation import get_input

# 事件分析
class Route:
    def __init__(self, route:[Point]):
        self.begin = route[0]
        self.end = route[len(route)-1]
        self.points = route
        self.count = len(route)

def departRoute(route:str):
    result = []
    num = 0
    for i in route:
        if i.isdecimal():
            num *= 10
            num += int(i)
        elif i == ' ' or i ==']':
            if num != 0:
                result.append(num)
            num = 0
    return result

def FindData_from_Points(p_id: int) -> Point:
    # make sql statement
    sqlStatement = "select * from Points where p_id=%d" % (p_id)
    print(sqlStatement)

    # execute sql statement
    gripData = accessDatabase(sqlStatement)

    # output data
    result = []
    for i in gripData:
        result.append(Point(id = i[0],
                            name = i[1],
                            long = i[2],
                            lati = i[3]))

    return result[0]

class RescuePlan:
    isFast = False
    def __init__(self, time:float, distance:float, vehicle_count:int, route:str):
        self.time = int(time * 60)  #时间向下取整
        self.distance = int(distance) #路程向下取整
        self.vehicle_count = vehicle_count

        Points_id = departRoute(route)
        Points = []
        for i in Points_id:
            Points.append(FindData_from_Points(p_id=i))
        self.route = Route(Points)


def make_rescuePlan(incident_id: list) -> [RescuePlan]:
    result = []
    carNums_Array, route_Array, distance_Array, times_Array = get_input(accident_index=incident_id)

    index = 0
    theShortTime = 0
    for i in range(0, len(carNums_Array) - 1):
        for j in range(0, len(carNums_Array[i]) - 1):
            plan = RescuePlan(distance=distance_Array[i][j],
                              time=times_Array[i][j],
                              vehicle_count=carNums_Array[i][j],
                              route=route_Array[i][j])
            result.append(plan)
            if result[theShortTime].time > plan.time:
                theShortTime = index
            index += 1
    result[theShortTime].isFast = True

    return result


