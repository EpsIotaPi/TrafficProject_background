
from dataBaseManage import accessDatabase

# 事件分析
class Incident:
    def __init__(self, id:int, groupName:str, content:str, type:str, status:str, updateTime:str,
                 times:str, highway_name:str, highway_num:str, road_section:str, direction:str, distance:str, position:int):
        self.id = id
        self.groupName = groupName
        self.content = content
        self.type = type
        self.status = status
        self.updateTime = updateTime
        self.time = times
        self.highway_name = highway_name
        self.highway_number = highway_num
        self.roadsection = road_section
        self.direction = direction
        self.distance = distance
        self.position = position

def FindData_from_Incidents(keyword:str, IncidentType:str) -> [Incident]:
    # make sql statement
    sqlStatement = "select * from Incidents"
    if keyword != '' or IncidentType != '':
        sqlStatement += ' where '
    if keyword != '':
        sqlStatement += "Content Like \'%" + keyword + "%\'"
        if IncidentType != '':
            sqlStatement += " and "
    if IncidentType != '':
        sqlStatement += "Incidents_Type=\'%s\'" % IncidentType
    print(sqlStatement)

    # execute sql statement
    gripData = accessDatabase(sqlStatement)

    # output data
    result = []
    for i in gripData:
        result.append(Incident(id=i[0],
                               groupName=i[1],
                               content=i[2],
                               type=i[3],
                               status=i[4],
                               updateTime=i[5],
                               times=i[6],
                               highway_name=i[7],
                               highway_num=i[8],
                               road_section=i[9],
                               direction=i[10],
                               distance=i[11],
                               position=i[12]))

    return result