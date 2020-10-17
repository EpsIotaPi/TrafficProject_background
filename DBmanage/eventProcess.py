
from DBmanage import *
from objectManage import *

def FindData_for_eventProcess(keyword = '', type = '', status = '已完成') -> [Incident]:
    """
    from Incidents Tabel
    """
    sqlBegin = "select * from Incidents where status = "
    sqlStatement = makeSql(sqlBegin, (status), '')

    if keyword != '':
        sqlStatement += ' and Content Like \"%' + keyword + '%\"'

    if type != '':
        sqlStatement += " and \"entity-label\" = \"%s\"" % type

    print(sqlStatement)

    gripData = accessDatabase(sqlStatement)

    result = []
    for i in gripData:
        en = Entity('')
        en.content = i[2]
        en.highway_name, en.highway_number, en.road_section, en.times = i[6], i[7], i[8], i[9]
        en.direction, en.distance, en.position, en.label = i[10], i[11], i[12], i[13]
        result.append(Incident(id=i[0],
                               serial_number=i[1],
                               content=i[2],
                               source=i[3],
                               response_level=i[4],
                               storage_time=i[5],
                               entity=en,
                               position=i[14],
                               status=i[15]))

    return result

