
from flask import Flask, request, jsonify

from dataBaseManage.ChatMonitor import *
from dataBaseManage.RescuePoints import *
from dataBaseManage.Incidents import *
from rescueDeployment.transportation import get_input


app = Flask(__name__)

# 群聊管理——ChatMonitor
@app.route('/groups')
def groups():
    keyword = request.args.get("keyword")
    status = request.args.get("status")
    if keyword == None:
        keyword = ''
    if status == None:
        status = ''

    groups = FindData_from_ChatMonitor(keyword=keyword, status=status)

    groupArray = []
    for i in groups:
        dict = {
            'name': i.name,
            'Introduction': i.Introduction,
            'owner': i.owner,
            'time': i.time,
            'status': i.status,
            'rate': int(i.entity_num / i.chat_num * 100)      #向下取整，可改进
        }
        groupArray.append(dict)

    # allGroup = FindData_from_ChatMonitor(keyword='', status='')

    outputData = {
        'code': 0,
        'message': '调用成功',
        'data': {
            'group_num': len(groupArray),
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
    keyword = request.args.get("keyword")
    label = request.args.get("label")
    if keyword == None:
        keyword = ''
    if label == None:
        label = ''

    events = FindData_from_Incidents(keyword, label)

    incidentArray = []
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
