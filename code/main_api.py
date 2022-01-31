## @package main
#  Main program for flask with routing logic and allowed methods
#
#  The main program routes the following routes under main prefix '/'.
#  As database use TinyDB, a json format database easy to use, view and
#  modify
#  All operations are logged into afc_dss.log file

import os
import sys
import logging
import logging.config
import loggly.handlers

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api

from api.env import secrets
from api import api_routes

from pathlib import Path
#  compatibility with flask run as api_classes is not in the same directory
#  as /flask/cly.py
cwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cwd)


# Configuracion del logger
# logging.config.fileConfig(Path('./api/env/loggly-dss.conf'))
# logger = logging.getLogger('WC-API')
# logger.info("API Arrancada")

app = Flask(__name__)

# # Configuracion de la base de datos
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////home/diego/weather-control/code/api/db/usuarios.db'
# db = SQLAlchemy(app)

# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)

#     def __repr__(self):
#         return '<User %r>' % self.username


# Configuracion de las rutas API
api = Api(app, prefix='/')
api.add_resource(api_routes.Prediccion, '/prediccion')

## TO RUN FROM FLASH: python3 main.py
app.run(host='0.0.0.0', port=secrets.PORT, debug=secrets.DEBUG)

## TO RUN WITH GUNICORN: gunicorn --bind 0.0.0.0:5000 main:app