
from DBmanage import *
from Date_Time import Time



def Find_position(id: int) -> (int, str):
    """

    :param id:
    :return: 第一项是position，第二项是serial_number
    """
    sqlStatement = "select position, serial_number from Incidents where incident_id=%d" % id

    gripData = accessDatabase(sqlStatement)

    list = []
    for i in gripData:
        list.append(i[0])
        list.append(i[1])

    return int(list[0]), str(list[1])


def Find_Incidents(status: str) -> [Time]:
    sqlStatement = "select storage_times from Incidents where status = "
    sqlStatement = makeSql(sqlStatement, status, "")
    gripData = accessDatabase(sqlStatement)
    result = []
    for i in gripData:
        print(i[0])
        result.append(Time(i[0]))
    return result

