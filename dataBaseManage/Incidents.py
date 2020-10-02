
from dataBaseManage import accessDatabase

# 事件分析
class Incident:
    def __init__(self, id:int, groupName:str, content:str, type:str, status:str, updateTime:str):
        self.id = id
        self.groupName = groupName
        self.content = content
        self.type = type
        self.status = status
        self.updateTime = updateTime

def FindData_from_Incidents(keyword:str, IncidentType:str) -> [Incident]:
    # make sql statement
    sqlStatement = "select * from Incidents where "
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
                               updateTime=i[5]))

    return result