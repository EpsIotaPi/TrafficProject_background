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

    def calInterval(self) -> (bool, bool, bool):
        """

        :return: (is_today, is_month, is_year)
        """
        now = datetime.datetime.now()
        is_year = (self.year == now.year)
        is_month = (is_year and self.month == now.month)
        is_today = (is_month and self.day == now.day)

        return is_today, is_month, is_year

    def __eq__(self, other):
        result = (self.year == other.year and self.month == other.month and self.day == other.day)
        return result




def somedays_ago(count:int) -> [Time]:
    """
    包括今天
    :param count:
    :return:
    """
    now = datetime.datetime.now()
    days = []
    for i in range(0, count):
        otherDay = now + datetime.timedelta(days = -i)
        days.append(Time(otherDay.strftime("%Y-%m-%d %H:%M:%S")))

    return days