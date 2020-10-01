from flask import Flask
from rescueDeployment.transportation import get_input
import numpy as np


app = Flask(__name__)


@app.route('/groups')
def groups():
    return 'Hello World!'

@app.route('/rescue_config')
def rescue_config():
    return 'Hello World!'

@app.route('/events')
def events():
    return 'Hello World!'

@app.route('/map')
def map():
    return 'Hello World!'

@app.route('/rescue')
def rescue():
    return 'Hello World!'


accident_index = np.array([21, 10, 17])  # 事故点的位置
a, b = get_input(accident_index)
print(np.array(a))
print(b)

if __name__ == '__app__':
    app.run()
