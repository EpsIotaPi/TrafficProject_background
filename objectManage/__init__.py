
from DBmanage import *
from Date_Time import *

from EntityRecog.predict import predict

class Entity:
    content = highway_name = highway_number = road_section = times = direction = distance = position = label = ""

    def __init__(self, content: str):
        self.content = content
        pos, dir, time, dis, rcst, hwn, hwnb, self.label = predict(content)
        self.position = self.__getDirContent(pos)
        self.direction = self.__getDirContent(dir)
        self.times = self.__getDirContent(time)
        self.distance = self.__getDirContent(dis)
        self.road_section = self.__getDirContent(rcst)
        self.highway_name = self.__getDirContent(hwn)
        self.highway_number = self.__getDirContent(hwnb)

    def __getDirContent(self, Dir):
        if len(Dir) > 0 and Dir[0] != '':
            return Dir[0]
        return ''

class Incident:
    def __init__(self, id, serial_number, content, source, response_level, storage_time, entity:Entity, position, status):
        self.id = id
        self.serial_number = serial_number
        self.content = content
        self.source = source
        self.response_level = response_level
        self.storage_time = storage_time
        self.position = position
        self.status = status
        self.entity = entity

class Scheme:
    def __init__(self, name:str, area:str, start_time:Time, end_time:Time, event_level:str, priority:str, description = '', id = 0):
        self.id = id
        self.name = name
        self.area = area
        self.start_time = start_time.outputStr()
        self.end_time = end_time.outputStr()
        self.event_level = event_level
        self.priority = priority
        self.description =  description

    def storge2DB(self):
        sql_begin = "Insert Into Schemes (name, area, start_time, end_time, event_level, priority, description) Values ("
        tup = (self.name, self.area, self.start_time, self.end_time, self.event_level, self.priority, self.description)
        sql_statement = makeSql(sql_begin, tup, ')')
        accessDatabase(sql_statement)

# class Highway:


class RescuePoint:
    def __init__(self, rp_name:str, admin_depart, contact_person, contact_number, ability, medical_depart, fire_depart, address, id = 0):
        self.id = id
        self.name = rp_name
        self.admin_depart = admin_depart
        self.contact_person = contact_person
        self.contact_number = contact_number
        self.ability = ability
        self.medical_depart = medical_depart
        self.fire_depart = fire_depart
        self.address = address

    def storge2DB(self):
        sql_begin = "Insert Into RescuePoints (rp_name, admin_depart, contact_person, contact_number, ability, medical_depart, fire_depart, address) Values ("
        tup = (self.name, self.admin_depart, self.contact_person, self.contact_number, self.ability, self.medical_depart, self.fire_depart, self.address)
        sql_statement = makeSql(sql_begin, tup, ')')
        accessDatabase(sql_statement)