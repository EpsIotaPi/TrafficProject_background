import sqlite3

conn = sqlite3.connect('TrafficDataBase.db')

print ("Opened database successfully")

cursor = conn.cursor()





cursor.close()
conn.commit()