
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
    return


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