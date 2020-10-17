
import os, sys
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)

import sqlite3


def accessDatabase(sqlStatement:str):
    # connect to database
    conn = sqlite3.connect('TrafficDB.db')
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



def makeSql(first_text: str, params: tuple, last_text:str) -> str:
    result = first_text
    if type(params) == tuple:
        for i in params:
            result += "\"" + str(i) + "\""
            if i != params[len(params) - 1]:
                result += ", "
    else:
        result += "\"" + str(params) + "\""
    result += last_text

    return result


