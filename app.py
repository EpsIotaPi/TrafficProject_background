from flask import Flask, request, jsonify

from dataClass import Incident
from dataBaseManage import FindData_from_Incidents
from rescueDeployment.transportation import get_input


app = Flask(__name__)


@app.route('/groups')
def groups():
    return 'Hello World!'

@app.route('/rescue_config')
def rescue_config():
    return 'Hello World!'

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
            'context': i.content,
            'label': i.type,
            'statue': i.status,
            'date': i.updateTime
        }
        incidentArray.append(dict)

    outputData = {
        'code': True,
        'message': '调用成功',
        'data': incidentArray
    }
    return jsonify(outputData)

@app.route('/map')
def map():
    return 'Hello World!'

@app.route('/rescue')
def rescue():
    return 'Hello World!'



if __name__ == '__app__':
    app.run()
