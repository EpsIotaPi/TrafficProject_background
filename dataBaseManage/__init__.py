
import os, sys
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)

import sqlite3
from dataClass import Incident


def FindData(keyword:str, IncidentType:str) -> [Incident]:
    # connect to database
    conn = sqlite3.connect('TrafficDataBase.db')
    print ("Opened database successfully")
    cursor = conn.cursor()

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
    gripData = cursor.execute(sqlStatement)

    # output data
    result = [Incident]
    for i in gripData:
        result.append(Incident(id=i[0],
                               groupName=i[1],
                               content=i[2],
                               type=i[3],
                               status=i[4],
                               updateTime=i[5]))

    # close connecting
    cursor.close()
    conn.commit()
    conn.close()
    return result


res = FindData('高速', '')
print(res[1].updateTime)

