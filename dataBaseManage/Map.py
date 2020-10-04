
import numpy as np
from dataBaseManage import *
from dataBaseManage.Incidents import *

def selectRandomAccidentPoints(num: int) -> [Point]:
    badPoints = {15, 19, 29, 33, 34, 45}
    baseSql = "select * from Points where p_id="

    result = []

    for i in range(0, num):
        p_id = 15
        while (p_id in badPoints):
            p_id = int(np.random.uniform(1, 46))
        badPoints.add(p_id)
        gripData = accessDatabase(baseSql + str(p_id))
        for j in gripData:
            result.append(Point(id=j[0],
                                name=j[1],
                                long=j[2],
                                lati=j[3]))
            print(len(result))
    return result

def countIncidents(Point) -> int:
    sqlStatement = "select * from Incidents where position_id=%d" % Point.id
    gripData = accessDatabase(sqlStatement)

    return len(gripData)




