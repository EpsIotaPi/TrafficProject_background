
from EntityRecog.predict import predict

defult = ''     #无信息时的默认值

def getDirContent(Dir):
    if len(Dir) > 0 and Dir[0] != defult:
        return Dir[0]
    return defult

def updataContent(Content, Dir):
    string = getDirContent(Dir)
    if string != defult:
        return string
    return Content

class entity_info:
    """输出内容"""
    def __init__(self, message):
        self.code = 1
        self.context = message

        self.position = self.direction = self.time = self.distance = self.road_section = self.highway_name = self.highway_number = defult
        if message != None and message != '':
            self.code = 1
            pos, dir, time, dis, rdsc, hwn, hwnb, type = predict(message)
            self.position = getDirContent(pos)
            self.direction = getDirContent(dir)
            self.time = getDirContent(time)
            self.distance = getDirContent(dis)
            self.road_section = getDirContent(rdsc)
            self.highway_name = getDirContent(hwn)
            self.highway_number = getDirContent(hwnb)
            self.type_info = type

#TODO:添加进数据库？

