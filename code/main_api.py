import os
import sys
import logging

from flask import Flask
from flask_restful import Api

from api import api_routes

#  compatibility with flask run as api_classes is not in the same directory
#  as /flask/cly.py
cwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cwd)

# Configuracion del logger
root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

app = Flask(__name__)

# Configuracion de las rutas API
api = Api(app, prefix='/')
api.add_resource(api_routes.Prediccion, '/prediccion')

port = os.environ.get('PREDICTION_PORT')
if port is None:
    port = 9001
   
## TO RUN FROM FLASH: python3 main.py
app.run(host='0.0.0.0', port=port, debug=True)
## TO RUN WITH GUNICORN: gunicorn --bind 0.0.0.0:5000 main:app