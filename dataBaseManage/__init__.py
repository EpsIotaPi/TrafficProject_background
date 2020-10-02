
import os, sys
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)

import sqlite3
from dataClass import *


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



