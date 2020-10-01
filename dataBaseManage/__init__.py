
import os, sys
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)

import sqlite3
from dataClass import Incident


def accessDatabase(sqlStatement:str):
    # connect to database
    conn = sqlite3.connect('TrafficDataBase.db')
    print ("Opened database successfully")
    cursor = conn.cursor()

    # execute sql statement and output data
    gripData = cursor.execute(sqlStatement)
    result = []
    for i in gripData:
        result.append(i)

    # close connecting
    cursor.close()
    conn.commit()
    conn.close()

    return result

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

def FindData_from():
    return 1


