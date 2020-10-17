
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