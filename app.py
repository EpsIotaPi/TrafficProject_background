from flask import Flask
from rescueDeployment.transportation import get_input
import numpy as np


app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__app__':
    print('hello')
    accident_index = np.array([21, 10, 17])  # 事故点的位置
    a, b = get_input(accident_index)
    print(np.array(a))
    print(b)
