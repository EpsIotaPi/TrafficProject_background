
from dataBaseManage import accessDatabase

# 群聊管理
class Group:
    def __init__(self, id: int, name:str, Introduction: str, owner: str, time: str, status: str, entity_num:int, chat_num:int):
        self.id = id
        self.name = name
        self.Introduction = Introduction
        self.owner = owner
        self.time = time
        self.status = status
        self.entity_num = entity_num
        self.chat_num = chat_num

def FindData_from_ChatMonitor(keyword:str, status:str) -> [Group]:
    # make sql statement
    sqlStatement = "select * from ChatMonitor where "
    if keyword != '':
        sqlStatement += "group_Name Like \'%" + keyword + "%\'"
        if status != '':
            sqlStatement += " and "
    if status != '':
        sqlStatement += "moni_Status=\'%s\'" % status

    # TODO: add ordby statement
    print(sqlStatement)

    # execute sql statement
    gripData = accessDatabase(sqlStatement)

    # output data
    result = []
    for i in gripData:
        result.append(Group(id=i[0],
                            name=i[1],
                            Introduction=i[2],
                            owner=i[3],
                            time=i[4],
                            status=i[5],
                            entity_num=i[6],
                            chat_num=i[7]))
    return result