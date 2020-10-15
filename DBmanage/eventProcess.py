
from DBmanage import accessDatabase
from objectManage import *

def FindData_for_eventProcess(keyword = '', type = '', status = 1) -> [Incident]:
    """
    from Incidents Tabel
    """
    sqlStatement = "select * from Incidents where status = %d" % status


    if keyword != '':
        sqlStatement += ' and Content Like \'%" + keyword + "%\''

    if type != '':
        sqlStatement += " and entity-label=\'%s\'" % type



    gripData = accessDatabase(sqlStatement)

    result = []
    for i in gripData:
        en = Entity(hwn=i[6],
                    hwnb=i[7],
                    rsct=i[8],
                    times=i[9],
                    direction=i[10],
                    distance=i[11],
                    position=i[12],
                    label=i[13])
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

