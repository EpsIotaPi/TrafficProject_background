import datetime


class Time:
    def __init__(self, text:str):
        self.year = int(text[0:4])
        self.month = int(text[5:7])
        self.day = int(text[8:10])
        self.hour = int(text[11:13])
        self.minute = int(text[14:16])
        self.second = int(text[17:19])


    def outputStr(self) -> str:
        return "%d-%02d-%02d %02d:%02d:%02d" % (self.year, self.month, self.day, self.hour, self.minute, self.second)

    def outputDateTime(self):
        return datetime.datetime(self.year, self.month, self.day,self.hour, self.minute, self.second)

    # 以分钟的形式返回
    def toNow(self):
        now = datetime.datetime.now()
        past = self.outputDateTime()
        return int((now - past).total_seconds() / 60)

    def toLastMonday(self):
        now = datetime.datetime.now()

        transDay = now.day
        if transDay == 0 and now.hour < 8:
            transDay = 7
        transHour = now.hour - 8
        transMin = now.minute
        transSec = now.second

        past = now - datetime.timedelta(days=transDay,
                                                hours=transHour,
                                                minutes=transMin,
                                                seconds=transSec)
        return int((now - past).total_seconds() / 60)


