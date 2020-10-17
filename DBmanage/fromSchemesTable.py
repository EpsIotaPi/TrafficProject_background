
from objectManage import Scheme
from DBmanage import *
from Date_Time import Time

def get_Scheme(keyword: str):
    sqlStatement = "select * from Schemes"
    if keyword != '':
        sqlStatement += " where name like \"%"
        sqlStatement = makeSql(sqlStatement, keyword, "%\"")

    print(sqlStatement)
    gripData = accessDatabase(sqlStatement)
    result = []
    for i in gripData:
        result.append(Scheme(id=i[0], name=i[1], area=i[2],
                             start_time=Time(i[3]), end_time=Time(i[4]),
                             event_level=i[5], priority=i[6], description=i[7]))

    return result
