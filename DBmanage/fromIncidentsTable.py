
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

def Find_source(source: str) -> [Time]:
    sqlStatement = "select status from Incidents where source = "
    sqlStatement = makeSql(sqlStatement, source, "")
    gripData = accessDatabase(sqlStatement)
    fixed = base = 0
    for i in gripData:
        if i[0] == '已完成':
            fixed += 1
        base += 1

    return base, fixed



def Update_Incidents_Status(id: int):
    begin_sql = "update Incidents set status = "
    end_sql = ' where incident_id = %d' % id
    sqlStatement = makeSql(begin_sql, '已确认', end_sql)

    accessDatabase(sqlStatement)