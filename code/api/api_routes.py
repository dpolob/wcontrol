# TODO Quitar logger mensajes de exceptions
# TODO Actualizar status

## @package ClassesAPI
#  Api code
#
#  Api implementation code

## @imports
# import errors
import datetime
import logging
from flask_restful import Resource
from importlib_metadata import metadata
# from json_manipulation import check_jsonkeys, StatusEnum, change_status
# from authorizations import ok_userpassword, requires_permission
# from flask_httpauth import HTTPBasicAuth
from tinydb import TinyDB, Query
from flask import request, jsonify, make_response, Response
from random import randint
# from mmt_thrift_client import MmtServiceSender

import requests
import json
import os, sys
# import globals
# import updateips
import jsonschema

# logger = logging.getLogger()

# # Enable HTTP Basic Authorization
# auth = HTTPBasicAuth()


# ## verify_password decorator
# #
# #  Override of method verify_password from HTTPBasicAuth package
# #  @param username: string. Got from HTTP header
# #  @param password: string. Got from HTTP header
# #  @return boolean
# @auth.verify_password
# def verify_password(username, password):
#     return ok_userpassword(username, password)

# ## Register class
# #
#  Implements code for Register API


from api.env import secrets
from api import api_modules
import time
import traceback

# logger = logging.getLogger()

class Prediccion(Resource):
       
    def get(self):
        token = "Token " + secrets.token_cesens
        now = datetime.datetime.now()
        past = now - datetime.timedelta(days=367) 
        
        ### PASADO
        try:
            dfs = api_modules.fetch_pasado(url=secrets.url_cesens, 
                                                 token=secrets.token_cesens, 
                                                 estaciones=secrets.estaciones_zona,
                                                 metricas=secrets.estaciones_metricas,
                                                 now=now,
                                                 past=past)
            df_pasado, metadata = api_modules.generar_variables_pasado(estaciones=dfs, 
                                                                       outliers=secrets.outliers_zona,
                                                                       pasado=secrets.pasado,
                                                                       now=now,
                                                                       escaladores=secrets.escaladores)
        except Exception as e:
            # logger.info(f"[Prediccion] No es posible conectar con Cesens. {e}")
            return Response(f"Problemas en la conexion con Cesens. {e, traceback.print_exc()}", status=500, mimetype='text/plain') 
        ### FUTURO
        try:
            
            df_futuro = api_modules.fetch_futuro(url=secrets.url_nwp, 
                                                 now=now.replace(minute=0, hour=0, second=0, microsecond=0))

            nwp = api_modules.generar_variables_futuro(df_futuro,
                                                       escaladores=secrets.escaladores)
        except Exception as e:
            # logger.info(f"[Prediccion] No es posible conectar con Cesens. {e}")
            return Response(f"Problemas en la conexion con NWP. {e, traceback.print_exc()}", status=500, mimetype='text/plain') 
            
        return Response(f"OK", status=200, mimetype='text/plain') 
                
                
        
    # response_cesens_dict = json.loads(response_cesens.text)
    # last_value = response_cesens_dict[list(response_cesens_dict)[len(response_cesens_dict) - 1]]

    # # check if current date differs from last date read
    # current_date = datetime.datetime.now()
    # current_date_ts = time.mktime(current_date.timetuple())
    # last_ts_read = int(list(response_cesens_dict)[len(response_cesens_dict) - 1])
    # if (current_date_ts - last_ts_read) > 24*60*60:
    #     save_errors.save_errors("WARNING: old data. More than one day")
    #     result['updated'] = "NO"
    # # calculations
    # cc = field_capacity(parameters=parameters)
    # pmp = wilting_point(parameters=parameters)
    # valor = (last_value - pmp) * (150) / 10 * \
    #     (1 - parameters['soil']['gross'] / 100) - (cc * (100 - parameters['water']['max']) / 100)

    # # valor puede ser negativo
    # if valor < 0:
    #     valor = cc - valor

    # # update parameters
    # parameters['last_check'] = int(list(response_cesens_dict)[len(response_cesens_dict) - 1])
    # save_parameters.save_parameters(parameters)

    # current_date = datetime.datetime.now()
    # current_date_ts = time.mktime(current_date.timetuple())

    # # Randomize
    # if parameters['randomize']:
    #     valor = valor - random.randrange(int(valor))

    # result[str(current_date_ts).split('.')[0]] = valor
    # return ("OK", result)
            
            
            
    #         db = TinyDB(globals.DATABASE)
    #         data = request.get_json()
    #         check_jsonkeys(data, 'Register')
            
    #         # Status: STARTED (requested for MMT), previously STOPPED
    #         data['status'] = StatusEnum.STARTED

    #         # add id a sequential number
    #         table= db.table('algorithm_list')  # switch to table
    #         list = table.all()

    #         #store basicAuth if exist
    #         user = request.args.get('user', default='entrypoint', type=str)
    #         password = request.args.get('password', default='fakepass', type=str)
    #         data['auth'] = {"user": user,
    #                         "password": password}

    #         data['id'] = randint(0,99999)

    #         # check if url_web is provided if not provide default url_web
    #         if 'urlweb' not in data.keys():
    #             logger.info("[Register API] Url web not provided. Used: {}".format(globals.DEFAULT_URL_WEB))
    #             data['urlweb'] = globals.DEFAULT_URL_WEB
            
    #         table= db.table('algorithm_list')  # switch to table

    #         if table.insert(data) < 0:
    #             raise errors.DBInsertionWrongException()
    #         logger.info("[Register API] Insetion in data base correct")
            
    #         logger.info("[Register API] Registration successful. Algorithm name: {} Id: {} Code 200 sent".format(data['name'], data['id']))
    #         return Response("Registration sucessful with id {}".format(data['id']), status=200, mimetype='text/plain')
        
    #     except (errors.DBInsertionWrongException):
    #         logger.info("[Register API][DBInsertionWrongException] Failure in registration. Code 500 sent")
    #         return Response("Registration failure", status=500, mimetype='text/plain')
    #     except (jsonschema.exceptions.ValidationError, jsonschema.exceptions.SchemaError) as error:
    #         logger.info('[Register API]Error in schemas validations {}'.format(error))
    #         return Response("Error in schema validation {}".format(error.message), status=500, mimetype='text/plain')
    #     except Exception as e:
    #         logger.info("[Register API][UncaughtException] {}".format(e))
    #         return Response("Registration failure. Check log", status=500, mimetype='text/plain')

