
from DBmanage import accessDatabase




def Find_position(id: int) -> (int,str):
    """

    :param id:
    :return: 第一项是position，第二项是serial_number
    """
    sqlStatement = "select position, serial_number from Incidents where id=%d" % id

    gripData = accessDatabase(sqlStatement)

    list = []
    for i in gripData:
        list.append(i[0])
        list.append(i[1])

    return int(list[0]), str(list[1])
