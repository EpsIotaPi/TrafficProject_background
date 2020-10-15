
from EntityRecog.predict import predict

class Entity:
    def __init__(self, content='', hwn='', hwnb='', rsct='', times='', direction='', distance='', position='', label=''):
        self.content = content
        self.highway_name = hwn
        self.highway_number = hwnb
        self.road_section = rsct
        self.times = times
        self.direction = direction
        self.distance = distance
        self.position = position
        self.label = label

    def __getDirContent(self, Dir):
        if len(Dir) > 0 and Dir[0] != '':
            return Dir[0]
        return ''

    def getData(self, content: str):
        self.content = content
        pos, dir, time, dis, rcst, hwn, hwnb, class_output = predict(content)
        self.position = self.__getDirContent(pos)
        self.direction = self.__getDirContent(dir)
        self.times = self.__getDirContent(time)
        self.distance = self.__getDirContent(dis)
        self.road_section = self.__getDirContent(rcst)
        self.highway_name = self.__getDirContent(hwn)
        self.highway_number = self.__getDirContent(hwnb)
        self.label = self.__getDirContent(class_output)


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