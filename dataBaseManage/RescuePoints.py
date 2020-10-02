
from dataBaseManage import accessDatabase

#
class coordinate:
    def __init__(self, longitude:float, latitude:float):
        self.longitude = longitude
        self.latitude = latitude

class Property:
    def __init__(self, TrafficPolice:int, RoadAdministration: int, small_ObstacleRemoval: int, large_ObstacleRemoval: int, Crane: int, BackTruck: int, PickupTruck: int, FireEngine: int, Ambulance: int):
        self.TrafficPolice = TrafficPolice
        self.RoadAdministration = RoadAdministration
        self.small_ObstacleRemoval = small_ObstacleRemoval
        self.large_ObstacleRemoval = large_ObstacleRemoval
        self.Crane = Crane
        self.BackTruck = BackTruck
        self.PickupTruck = PickupTruck
        self.FireEngine = FireEngine
        self.Ambulance = Ambulance

class Point:
    def __init__(self, id: int, name:str, long: float, lati:float):
        self.id = id
        self.name = name
        self.position = coordinate(long, lati)

class rescuePoint(Point):
    def __init__(self, id: int, name:str, long: float, lati:float, property:Property):
        Point.__init__(self, id, name, long, lati)
        self.property = property



def FindData_from_RescuePoints(keyword:str) -> [rescuePoint]:
    # make sql statement
    sqlStatement = "select * from RescuePoints"
    if keyword != '':
        sqlStatement += " where rp_name Like \'%" + keyword + "%\'"

    print(sqlStatement)

    # execute sql statement
    gripData = accessDatabase(sqlStatement)

    # output data
    result = []
    for i in gripData:
        property = Property(TrafficPolice=i[2],
                            RoadAdministration=i[3],
                            small_ObstacleRemoval=i[4],
                            large_ObstacleRemoval=i[5],
                            Crane=i[6],
                            BackTruck=i[7],
                            PickupTruck=i[8],
                            FireEngine=i[9],
                            Ambulance=i[10])
        result.append(rescuePoint(id=i[0],
                                  name=i[1],
                                  # 本页面不需要经纬度信息
                                  long=120,
                                  lati=120,
                                  property=property,))
    return result