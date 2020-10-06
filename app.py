
from flask import Flask, request, jsonify
from flask_cors import *
import json, datetime, time


from dataBaseManage.ChatMonitor import *
from dataBaseManage.RescuePoints import *
from dataBaseManage.Incidents import *
from dataBaseManage.Map import *
from dataBaseManage.RescuePlan import *

# ---------------------------------------------------
app = Flask(__name__)
CORS(app, supports_credentials=True)

# ---------------------------------------------------
def pageManage(pageNum: int, index:int, info_count:int) -> bool:
    start = (pageNum - 1) * info_count
    end = pageNum * info_count - 1
    if index >= start and index <= end:
        return True
    return False

def calTimes():
    nowTime = time.localtime()
    transDay = nowTime.tm_wday
    if transDay == 0 and nowTime.tm_hour < 8:
        transDay = 7
    transHour = nowTime.tm_hour - 8
    transMin = nowTime.tm_min
    transSec = nowTime.tm_sec

    nowTime = datetime.datetime.now()
    startDay = nowTime - datetime.timedelta(days=transDay,
                                            hours=transHour,
                                            minutes=transMin,
                                            seconds=transSec)
    transTime = int((nowTime - startDay).total_seconds() / 60)
    return transTime

# ---------------------------------------------------


# 群聊管理——ChatMonitor
@app.route('/groups')
def groups():
    pageNum = request.args.get('page_number')
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
            'rate': int(i.entity_num / i.chat_num * 100)
        }
        if pageManage(pageNum=int(pageNum), index=index,info_count=10):
            groupArray.append(dict)
        index += 1

    allGroup = FindData_from_ChatMonitor(keyword='', status='')

    allEntity = 0
    allGroupNum = 0
    for i in allGroup:
        allGroupNum += 1
        allEntity += i.entity_num

    outputData = {
        'code': 1,
        'message': '调用成功',
        'data': {
            'group_num': allGroupNum,
            'time': calTimes(),
            'events_num': allEntity,
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
        if pageManage(pageNum=int(pageNum), index=index, info_count=10):
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
@app.route('/events')
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
            'updateTime': i.updateTime,
            'entity_info':{
                'time': i.time,
                'highway_name': i.highway_name,
                'highway_numbre': i.highway_number,
                'road_section': i.roadsection,
                'direction': i.direction,
                'distance': i.distance,
                'p_id':i.position
            }
        }
        if pageManage(pageNum=int(pageNum), index=index, info_count=10):
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
    num = int(request.args.get('need_num'))
    choosePoints = selectRandomAccidentPoints(num)
    accidentPointsArray = []
    for i in choosePoints:
        dict = {
            "name": i.name,
            "incident_num": countIncidents(i),
            "coordinate": {
                "long": i.position.longitude,
                "lati": i.position.latitude
            }
        }
        accidentPointsArray.append(dict)

    outputData = {
        'code': 1,
        'message': '调用成功',
        'data': {
            'accidentPoints': accidentPointsArray
        }
    }

    return jsonify(outputData)


# 事故救援——Points
@app.route('/rescue', methods=["GET", "POST"])
def rescue():
    p_id = json.loads(request.get_data(as_text=True))

    Plans = make_rescuePlan(np.array(p_id))

    incidentArray = []
    for i in Plans:
        routeArray = []
        for p in i.route.points:
            point = {
                'point_id': p.id,
                'point_name': p.name,
                'coordinate': {
                    'long':p.position.longitude,
                    'lati':p.position.latitude
                }
            }
            routeArray.append(point)
        dict = {
            'begin': i.route.begin.id,
            'end': i.route.end.id,
            'time': i.time,
            'distance': i.distance,
            'vehicle_count': i.vehicle_count,
            'route': routeArray,
            'isFast': i.isFast
        }
        incidentArray.append(dict)

    outputData = {
        'code': 1,
        'message': '调用成功',
        'data': {
            'incident': incidentArray
        }
    }

    return jsonify(outputData)



if __name__ == '__main__':

    app.run()
