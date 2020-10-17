
from flask import Flask, request, jsonify
from flask_cors import *
from objectManage import *
from DBmanage.eventProcess import FindData_for_eventProcess
from DBmanage.fromIncidentsTable import *
import map

from rescueDeployment import *
from rescueDeployment.transportation import rescuePlan_API

# from EntityRecog.predict import predict

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

def get_args(param:str):
    result = request.args.get(param)
    if result == None:
        result = ''
    return result

# ---------------------------------------------------

# new version

# 系统菜单
@app.route('/system_menu')
def sysMenu():

    iDay = iMon = iYear = rDay = rMon = rYear = 0
    finished = Find_Incidents('已完成')
    checked = Find_Incidents('已确认')
    unchecked = Find_Incidents('待确认')

    for i in finished:
        print(i.calInterval())
        if i.calInterval()[0]:
            rDay += 1
            iDay += 1
        if i.calInterval()[1]:
            rMon += 1
            iMon += 1
        if i.calInterval()[2]:
            rYear += 1
            iYear += 1
    for i in checked:
        if i.calInterval()[0]:
            iDay += 1
        if i.calInterval()[1]:
            iMon += 1
        if i.calInterval()[2]:
            iYear += 1
    for i in unchecked:
        if i.calInterval()[0]:
            iDay += 1
        if i.calInterval()[1]:
            iMon += 1
        if i.calInterval()[2]:
            iYear += 1

    staticArray = []
    day_list = somedays_ago(13)

    for day in day_list:
        finish = connect = 0

        for i in finished:
            if i == day:
                finish += 1
                connect += 1
        for i in checked:
            if i == day:
                connect += 1
        for i in unchecked:
            if i == day:
                connect += 1

        dic = {
            'name': day.outputDateTime().strftime("%m-%d"),
            'value': {
                'finish_incidents': finish,
                'connect_incidents': connect
            }
        }
        staticArray.append(dic)

    outputData = {
        'code': 1,
        'message': '调用成功',
        'data': {
            'incident': {
                'day': iDay,
                'month': iMon,
                'year': iYear
            },
            'rescue': {
                'day': rDay,
                'month': rMon,
                'year': rYear
            },
            'static': staticArray
        }
    }

    return jsonify(outputData)


@app.route('/event_process')
def eventProcess():
    keyword = get_args('keyword')
    incident_type = get_args('type')
    if keyword == None:
        keyword = ''
    if incident_type == None:
        incident_type = ''
    page_num = int(request.args.get('page'))
    status = request.args.get('status')

    dataArray = FindData_for_eventProcess(keyword, incident_type, status)

    resultData = []
    for i in range(0, len(dataArray)):
        if pageManage(pageNum=page_num, index=i, info_count=10):
            pastTime = Time(dataArray[i].storage_time)
            obj = {
                "id": dataArray[i].id,
                "source": dataArray[i].source,
                "lv": dataArray[i].response_level,
                "serial_num": dataArray[i].serial_number,
                "highway_name": dataArray[i].entity.highway_name,
                "road_section": dataArray[i].entity.road_section,
                "direction": dataArray[i].entity.direction,
                "time": dataArray[i].entity.times,
                "duration": pastTime.toNow(),
                "label": dataArray[i].entity.label,
                "point": dataArray[i].position,
                "content": dataArray[i].content
            }
            resultData.append(obj)

    outputData = {
        'code': 1,
        'message': '调用成功',
        'data': resultData
    }

    return jsonify(outputData)


@app.route('/route_recommend')
def routeRecommend():
    incidentID_get = request.args.getlist("incident_id")
    pos_id = []
    serial = []
    for i in incidentID_get:
        Pos, Ser = Find_position(int(i))
        pos_id.append(Pos)
        serial.append(Ser)

    rescuePlans = allPlans(rescuePlan_API(pos_id))
    rescuePlans.self_compare()
    rescuePlans.giveSerial(pos_id, serial)

    map.clear()
    for id in pos_id:
        map.randomTraffic(id, 100)
    map.uniform(len(pos_id))


    rescuePlansArray = []
    for plan in rescuePlans.plan_list:

        incidentArray = []
        for path in plan.path_list:
            routeArray = []
            for node in path.route_node:
                route_dic = {
                    'point_name': node.name,
                    'point_id': node.id
                }
                routeArray.append(route_dic)

            incident_dic = {
                'rescue_routes': routeArray,
                'incident_serial': path.serial_number,
                'car_num': path.carNum,
                'rescue_time': int(path.time * 60),
                'rescue_distance': path.distance,
                'congestion_rate': path.calCongestionRate()
            }
            incidentArray.append(incident_dic)

        rescuePlan_dic = {
            'incidents': incidentArray,
            'joint_time': int(plan.sum_time * 60),
            'joint_distance': plan.sum_distance,
            'compare_average_time': plan.compare_avgTime,
            'compare_average_distance': plan.compare_avgDis,
            'is_fast': plan.is_fast,
            'is_short': plan.is_short,
        }
        rescuePlansArray.append(rescuePlan_dic)

    outputData = {
        'code': 1,
        'message': "调用成功",
        'data': {
            'rescuePlans': rescuePlansArray,
            'average_time': int(rescuePlans.avgTime * 60),
            'average_distance': rescuePlans.avgDis,
            'rescue_incidents': serial
        }
    }
    return jsonify(outputData)

@app.route('/plan')
def showScheme():
    keyword = get_args('keyword')
    page_num = get_args('page')

    schemeArray = []
    scheme_list = []

    for i in range(0, len(schemeArray)):
        if pageManage(pageNum=page_num, index=i, info_count=10):
            scheme = schemeArray[i]
            dic = {
                'id': scheme.id,
                'name': scheme.name,
                'area': scheme.area,
                'lv': scheme.event_level,
                'create_person': '',
                'create_time': '',
                'status': '未启用'
            }
            scheme_list.append(dic)

    outputData = {
        'code': 1,
        'message': "调用成功",
        'data': scheme_list
    }
    return jsonify(outputData)


@app.route('/create_plan')
def createScheme():
    scheme_name = get_args('name')
    scheme_area = get_args('area')
    start_time = Time(get_args('start_time'))
    end_time = Time(get_args('end_time'))
    event_level = get_args('event_lv')
    priority = get_args('priority')
    description = get_args('description')

    newScheme = Scheme(scheme_name, scheme_area, start_time, end_time, event_level, priority, description)
    newScheme.storge2DB()

    outputData = {
        'code': 1,
        'message': "调用成功",
    }
    return jsonify(outputData)



# # old version
# # 群聊管理——ChatMonitor
# @app.route('/groups')
# def groups():
#     pageNum = request.args.get('page_number')
#     keyword = request.args.get("keyword")
#     status = request.args.get("status")
#     if keyword == None:
#         keyword = ''
#     if status == None:
#         status = ''
#
#     groups = FindData_from_ChatMonitor(keyword=keyword, status=status)
#
#     groupArray = []
#     index = 0
#     for i in groups:
#         dict = {
#             'name': i.name,
#             'Introduction': i.Introduction,
#             'owner': i.owner,
#             'time': i.time,
#             'status': i.status,
#             'rate': int(i.entity_num / i.chat_num * 100)
#         }
#         if pageManage(pageNum=int(pageNum), index=index,info_count=10):
#             groupArray.append(dict)
#         index += 1
#
#     allGroup = FindData_from_ChatMonitor(keyword='', status='')
#
#     allEntity = 0
#     allGroupNum = 0
#     for i in allGroup:
#         allGroupNum += 1
#         allEntity += i.entity_num
#
#     outputData = {
#         'code': 1,
#         'message': '调用成功',
#         'data': {
#             'group_num': allGroupNum,
#             'time': calTimes(),
#             'events_num': allEntity,
#             'group_info': groupArray
#         }
#     }
#     return jsonify(outputData)
#
#
# #在线录入
# @app.route('/add_new')
# def add_new():
#     message = request.args.get("context")
#     if message == None:
#         message = ''
#
#     Incident = entity_info(message)
#
#
#     outputData = {
#         'code': 1,
#         'message': '调用成功',
#         'data': {
#             'Incident_time': Incident.time,
#             'Incident_type': Incident.type_info,
#             'highway_name': Incident.highway_name,
#             'highway_num': Incident.highway_number,
#             'highway_direction': Incident.direction,
#             'rode_section': Incident.road_section,
#             'distance': Incident.distance
#         }
#     }
#
#     return jsonify(outputData)
#
#
# # 救援点配置——RescuePoints
# @app.route('/rescue_config')
# def rescue_config():
#     keyword = request.args.get("keyword")
#     pageNum = request.args.get("page_num")
#     if keyword == None:
#         keyword = ''
#
#     rescuePoints = FindData_from_RescuePoints(keyword)
#
#     rp_Array = []
#     index = 0
#     for i in rescuePoints:
#         dict = {
#             'rp_name': i.name,
#             'rp_property':{
#                 'TrafficPolice': i.property.TrafficPolice,
#                 'RoadAdministration': i.property.RoadAdministration,
#                 'small_ObstacleRemoval': i.property.small_ObstacleRemoval,
#                 'large_ObstacleRemoval': i.property.large_ObstacleRemoval,
#                 'Crane': i.property.Crane,
#                 'BackTruck': i.property.BackTruck,
#                 'PickupTruck': i.property.PickupTruck,
#                 'FireEngine': i.property.FireEngine,
#                 'Ambulance': i.property.Ambulance
#             }
#         }
#         if pageManage(pageNum=int(pageNum), index=index, info_count=10):
#             rp_Array.append(dict)
#
#     outputData = {
#         'code': 1,
#         'message': '调用成功',
#         'data': {
#             'rp_num': len(rp_Array),
#             'rescuePoint': rp_Array
#         }
#     }
#
#     return jsonify(outputData)
#
#
#
# # 实时地图
# @app.route('/map')
# def map():
#     num = int(request.args.get('need_num'))
#     choosePoints = selectRandomAccidentPoints(num)
#     accidentPointsArray = []
#     for i in choosePoints:
#         dict = {
#             "name": i.name,
#             "incident_num": countIncidents(i),
#             "entity_info": {
#                 "p_id": i.id
#             },
#             "coordinate": {
#                 "long": i.position.longitude,
#                 "lati": i.position.latitude
#             }
#         }
#         accidentPointsArray.append(dict)
#
#     outputData = {
#         'code': 1,
#         'message': '调用成功',
#         'data': {
#             'accidentPoints': accidentPointsArray
#         }
#     }
#     return jsonify(outputData)
#
#
# # 事故救援——Points
# @app.route('/rescue')
# def rescue():
#     pid_get = request.args.getlist("idList[]")
#     p_id = []
#     for i in pid_get:
#         p_id.append(int(i))
#     Plans = make_rescuePlan(np.array(p_id))
#
#     incidentArray = []
#     for i in Plans:
#         routeArray = []
#         for p in i.route.points:
#             point = {
#                 'point_id': p.id,
#                 'point_name': p.name,
#                 'coordinate': {
#                     'long':p.position.longitude,
#                     'lati':p.position.latitude
#                 }
#             }
#             routeArray.append(point)
#         dict = {
#             'begin': i.route.begin.id,
#             'end': i.route.end.id,
#             'time': i.time,
#             'distance': i.distance,
#             'vehicle_count': i.vehicle_count,
#             'route': routeArray,
#             'isFast': i.isFast
#         }
#         incidentArray.append(dict)
#
#     outputData = {
#         'code': 1,
#         'message': '调用成功',
#         'data': {
#             'incident': incidentArray
#         }
#     }
#     return jsonify(outputData)



if __name__ == '__main__':

    app.run()
