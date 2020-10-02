
from flask import Flask, request, jsonify

from dataBaseManage.ChatMonitor import *
from dataBaseManage.RescuePoints import *
from dataBaseManage.Incidents import *
from rescueDeployment.transportation import get_input

# ---------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------
def pageManage(pageNum: int, index:int, info_count:int) -> bool:
    start = (pageNum - 1) * info_count
    end = pageNum * info_count - 1
    if index >= start and index <= end:
        return True
    return False


# ---------------------------------------------------


# 群聊管理——ChatMonitor
@app.route('/groups')
def groups():
    pageNum = request.args.get('page_num')
    keyword = request.args.get("keyword")
    status = request.args.get("status")
    if keyword == None:
        keyword = ''
    if status == None:
        status = ''

    groups = FindData_from_ChatMonitor(keyword=keyword, status=status)

    groupArray = []
    index = 0
    for i in groups:
        dict = {
            'name': i.name,
            'Introduction': i.Introduction,
            'owner': i.owner,
            'time': i.time,
            'status': i.status,
            'rate': int(i.entity_num / i.chat_num * 100)      #向下取整，可改进
        }
        if pageManage(pageNum=int(pageNum), index=index,info_count=5):
            groupArray.append(dict)
        index += 1

    allGroup = FindData_from_ChatMonitor(keyword='', status='')

    outputData = {
        'code': 0,
        'message': '调用成功',
        'data': {
            'group_num': len(allGroup),
            'time': 87,
            'events_num': 3000,
            'group_info': groupArray
        }
    }
    return jsonify(outputData)


# 救援点配置——RescuePoints
@app.route('/rescue_config')
def rescue_config():
    keyword = request.args.get("keyword")
    pageNum = request.args.get("page_num")
    if keyword == None:
        keyword = ''

    rescuePoints = FindData_from_RescuePoints(keyword)

    rp_Array = []
    index = 0
    for i in rescuePoints:
        dict = {
            'rp_name': i.name,
            'rp_property':{
                'TrafficPolice': i.property.TrafficPolice,
                'RoadAdministration': i.property.RoadAdministration,
                'small_ObstacleRemoval': i.property.small_ObstacleRemoval,
                'large_ObstacleRemoval': i.property.large_ObstacleRemoval,
                'Crane': i.property.Crane,
                'BackTruck': i.property.BackTruck,
                'PickupTruck': i.property.PickupTruck,
                'FireEngine': i.property.FireEngine,
                'Ambulance': i.property.Ambulance
            }
        }
        if pageManage(pageNum=int(pageNum), index=index, info_count=9):
            rp_Array.append(dict)

    outputData = {
        'code': 1,
        'message': '调用成功',
        'data': {
            'rp_num': len(rp_Array),
            'rescuePoint': rp_Array
        }
    }

    return jsonify(outputData)


# 事件分析——Incidents
@app.route('/events', methods=["GET", "POST"])
def events():
    pageNum =  request.args.get("page_num")
    keyword = request.args.get("keyword")
    type = request.args.get("type")
    if keyword == None:
        keyword = ''
    if type == None:
        type = ''

    events = FindData_from_Incidents(keyword, type)


    incidentArray = []
    index = 0
    for i in events:
        dict = {
            'name': i.groupName,
            'content': i.content,
            'type': i.type,
            'status': i.status,
            'updateTime': i.updateTime
            # TODO:add entity_info
            # 'entity_info':
        }
        if pageManage(pageNum=int(pageNum), index=index, info_count=5):
            incidentArray.append(dict)

    outputData = {
        'code': 1,
        'message': '调用成功',
        'data': incidentArray
    }
    return jsonify(outputData)


# 实时地图
@app.route('/map')
def map():
    return 'Hello World!'


# 事故救援
@app.route('/rescue')
def rescue():
    return 'Hello World!'



if __name__ == '__app__':
    app.run()
