
from objectManage import *
from DBmanage import *
from Date_Time import Time

def get_Scheme(keyword: str):
    sqlStatement = "select * from Schemes"
    if keyword != '':
        sqlStatement += " where name like \"%" + keyword + "%\""

    print(sqlStatement)
    gripData = accessDatabase(sqlStatement)
    result = []
    for i in gripData:
        result.append(Scheme(id=i[0], name=i[1], area=i[2], create_time=Time(i[3]),
                             event_level=i[4], priority=i[5], description=i[6]))

    return result


def get_highway(keyword: str):
    sqlStatement = "select * from highways"
    if keyword != '':
        sqlStatement += " where name like \"%" + keyword + "%\""

    print(sqlStatement)
    gripData = accessDatabase(sqlStatement)
    result = []
    for i in gripData:
        result.append(Highway(id=i[0], name=i[1], entrance=i[2], bridge_tunnel=i[3],
                              manage_office=i[4], stake=i[5], block=i[6], accident=i[7],
                              rescue_point=i[8], engin=i[9], speed_limit=i[10], weather=i[11]))

    return result


def get_rescuePoint(keyword: str) -> [RescuePoint]:
    sqlStatement = "select * from RescuePoints"
    if keyword != '':
        sqlStatement += " where name like \"%" + keyword + "%\""

    gripData = accessDatabase(sqlStatement)
    result = []
    for i in gripData:
        result.append(RescuePoint(id=i[0], rp_name=i[1], admin_depart=i[2],
                                  contact_person=i[3], contact_number=i[4],
                                  ability=i[5], medical_depart=i[6], fire_depart=i[7], address=i[8]))

    return result


def get_groups(keyword:str, status:str) -> [Group]:
    # make sql statement
    sqlStatement = "select * from groups"
    if keyword != '' or status != '':
        sqlStatement += ' where '
    if keyword != '':
        sqlStatement += "group_Name Like \'%" + keyword + "%\'"
        if status != '':
            sqlStatement += " and "
    if status != '':
        sqlStatement += "moni_Status=\'%s\'" % status


    # execute sql statement
    gripData = accessDatabase(sqlStatement)

    # output data
    result = []
    for i in gripData:
        result.append(Group(id=i[0], name=i[1], source=i[2],
                            owner=i[3], time=i[4], status=i[5],
                            entity_num=i[6], chat_num=i[7]))
    return result