
from flask import Flask, request, jsonify
from flask_cors import *
from objectManage import *
from DBmanage.eventProcess import FindData_for_eventProcess
from DBmanage.fromIncidentsTable import *
from DBmanage.fromListTable import *
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
    incident_id = int(get_args('incident_id'))
    if incident_id != 0:
        Update_Incidents_Status(incident_id)

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
                'rescue_distance': int(path.distance),
                'congestion_rate': path.calCongestionRate()
            }
            incidentArray.append(incident_dic)

        rescuePlan_dic = {
            'incidents': incidentArray,
            'joint_time': int(plan.sum_time * 60),
            'joint_distance': int(plan.sum_distance),
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
            'average_distance': int(rescuePlans.avgDis),
            'rescue_incidents': serial
        }
    }
    return jsonify(outputData)

@app.route('/plan')
def showScheme():
    keyword = get_args('keyword')
    page_num = int(get_args('page'))

    schemeArray = get_Scheme(keyword)
    scheme_list = []

    for i in range(0, len(schemeArray)):
        if pageManage(pageNum=page_num, index=i, info_count=10):
            scheme = schemeArray[i]
            dic = {
                'id': scheme.id,
                'name': scheme.name,
                'area': scheme.area,
                'lv': scheme.event_level,
                # TODO: need a create_person
                'create_person': "张三",
                'create_time': scheme.create_time.outputStr(),
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
    event_level = get_args('event_lv')
    priority = get_args('priority')
    description = get_args('description')

    newScheme = Scheme(scheme_name, scheme_area, event_level, priority, description)
    newScheme.storge2DB()

    outputData = {
        'code': 1,
        'message': "调用成功",
    }
    return jsonify(outputData)


@app.route('/highway')
def showHighway():
    keyword = get_args('keyword')
    page_num = get_args('page_num')

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


@app.route('/rescue_point')
def show_resucePoint():
    keyword = get_args('keyword')
    page_num = int(get_args('page_num'))

    rpArray = get_rescuePoint(keyword)
    rp_list = []

    for i in range(0, len(rpArray)):
        if pageManage(pageNum=page_num, index=i, info_count=10):
            rp = rpArray[i]
            dic = {
                'id': rp.id,
                'rp_name': rp.name,
                'admin_depart': rp.admin_depart,
                'contact_person': rp.contact_person,
                'contact_number': rp.contact_number,
                'ability': rp.ability,
                'medical_depart': rp.medical_depart,
                'fire_depart': rp.fire_depart,
                'address': rp.address
            }
            rp_list.append(dic)

    outputData = {
        'code': 1,
        'message': "调用成功",
        'data': rp_list
    }
    return jsonify(outputData)


@app.route('/create_rp')
def createRP():
    rp_name = get_args('rp_name')
    admin_depart = get_args('admin_depart')
    contact_person = get_args('contact_person')
    contact_number = get_args('contact_number')
    address = get_args('address')
    ability = get_args('ability')
    medical_depart = get_args('medical_depart')
    fire_depart = get_args('fire_depart')

    newRP = RescuePoint(rp_name=rp_name, admin_depart=admin_depart, contact_person=contact_person, contact_number=contact_number,
                        address=address, ability=ability, medical_depart=medical_depart,fire_depart=fire_depart)
    newRP.storge2DB()

    outputData = {
        'code': 1,
        'message': "调用成功",
    }
    return jsonify(outputData)


@app.route('/groups')
def groups():
    pageNum = get_args('page_number')
    keyword = get_args("keyword")
    status = get_args("status")


    groups = get_groups(keyword=keyword, status=status)

    groupArray = []
    index = 0
    for i in groups:
        dict = {
            'name': i.name,
            'source': i.source,
            'owner': i.owner,
            'time': i.time,
            'status': i.status,
            'rate': int(i.entity_num / i.chat_num * 100)
        }
        if pageManage(pageNum=int(pageNum), index=index,info_count=10):
            groupArray.append(dict)
        index += 1

    allGroup = get_groups(keyword='', status='')

    allEntity = 0
    allGroupNum = 0
    for i in allGroup:
        allGroupNum += 1
        allEntity += i.entity_num

    wx_base, wx_fixed = Find_source("微信群聊")
    zhu_base, zhu_fixed = Find_source("智慧云平台")
    text_base, text_fixed = Find_source("在线录入")

    outputData = {
        'code': 1,
        'message': '调用成功',
        'data': {
            'group_num': allGroupNum,
            'events_num': allEntity,
            'group_info': groupArray,
            'base_data': {
                'WX': {
                    'base': wx_base,
                    'fixed': wx_fixed,
                    'rate': int(wx_fixed * 100 / wx_base),
                },
                'ZHU': {
                    'base': zhu_base,
                    'fixed': zhu_fixed,
                    'rate': int(zhu_fixed * 100 / zhu_base),
                },
                'TEXT': {
                    'base': text_base,
                    'fixed': text_fixed,
                    'rate': int(text_fixed * 100 / text_base),
                }
            }
        }
    }
    return jsonify(outputData)


if __name__ == '__main__':
    app.run()
