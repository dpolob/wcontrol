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

import torch
import numpy as np
# logger = logging.getLogger()
from pathlib import Path
from common.utils.trainers import trainerSeq2Seq as tr
from common.utils.trainers import trainerpmodel as tr_pmodel
from common.utils.scalers import Escalador

class Prediccion(Resource):
       
    def get(self):
        token = "Token " + secrets.token_cesens
        now = datetime.datetime.now()
        past = now - datetime.timedelta(days=45) 
        
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
                                                                       escaladores=secrets.escaladores,
                                                                       cdg=secrets.CdG)

        except Exception as e:
            # logger.info(f"[Prediccion] No es posible conectar con Cesens. {e}")
            return Response(f"Problemas en la conexion con Cesens. {e}", status=500, mimetype='text/plain') 
        ### FUTURO
        try:
            df_futuro = api_modules.fetch_futuro(url=secrets.url_nwp, 
                                                 now=now.replace(minute=0, hour=0, second=0, microsecond=0),
                                                 future=72)
            nwp = api_modules.generar_variables_futuro(df_futuro,
                                                    estaciones=df_pasado,
                                                       escaladores=secrets.escaladores)
        except Exception as e:
            # logger.info(f"[Prediccion] No es posible conectar con Cesens. {e}")
            return Response(f"Problemas en la conexion con NWP. {e}", status=500, mimetype='text/plain') 
        
        
        ## PREDICCION ZONAL
        try:
            
            Xf, Yt, P = api_modules.fff(datasets=df_pasado,nwps=nwp, pasado=secrets.pasado, futuro=72, etiquetaF=secrets.Ff, etiquetaT=secrets.Ft, etiquetaP=secrets.Fnwp)
                    
            print(f"{Xf.shape=}")
            print(f"{Yt.shape=}")
            print(f"{P.shape=}")
            
            device = secrets.device
            path_zmodel_checkpoints = secrets.path_zmodel_checkpoints
            path_zmodel = secrets.path_zmodel
            model = torch.load(Path(path_zmodel) , map_location='cpu')
            model.to(device)

            trainer = tr.TorchTrainer(model=model,
                                    device=device,
                                    checkpoint_folder= path_zmodel_checkpoints)
            trainer._load_best_checkpoint()
            y_pred = trainer.predict_one(Xf, Yt, P)

            print(f"{y_pred.shape=}")
            
        
            # y_t = np.empty(shape=(y_pred.shape[1], 1))  # N,Ly,Fout -> (Ly,1)
            # y_hr =np.empty_like(y_t)
            # y_rain=np.empty(shape=(y_pred.shape[1], 8))
            
            y_t = torch.mean(y_pred[..., 0], dim=0)
            y_hr = torch.mean(y_pred[..., 1], dim=0)
            y_rain = torch.mean(y_pred[..., 2:], dim=0)
            y_t = torch.unsqueeze(y_t, dim=-1)
            y_hr = torch.unsqueeze(y_hr, dim=-1)
            print(f"{y_t.shape=}")
            print(f"{y_hr.shape=}")
            print(f"{y_rain.shape=}")
            y_pred_zmodel = torch.cat([y_t,y_hr,y_rain], dim=-1)
            y_pred_zmodel = torch.unsqueeze(y_pred_zmodel, dim=0)
            print(f"{y_pred_zmodel.shape=}")
            
            
        except Exception as e:
            # logger.info(f"[Prediccion] No es posible conectar con Cesens. {e}")
            return Response(f"Problemas en la prediccion zonal. {e}", status=500, mimetype='text/plain') 
            
        ## ZMODELO TEMPERATURA
        try:
            device = secrets.device
            path_temp_checkpoints = secrets.path_temp_checkpoints
            path_temp_model = secrets.path_temp_model
            model = torch.load(Path(path_temp_model) , map_location='cpu')
            model.to(device)
            
            trainer = tr_pmodel.TorchTrainer(model=model,
                                            device=device,
                                            checkpoint_folder= path_temp_checkpoints)
            trainer._load_best_checkpoint()
            y_pred_temp = trainer.predict_one(torch.unsqueeze(y_pred_zmodel[...,0], dim=-1))
            y_pred_temp = torch.squeeze(y_pred_temp)
            print(f"{y_pred_temp.shape=}")
        except Exception as e:
            # logger.info(f"[Prediccion] No es posible conectar con Cesens. {e}")
            return Response(f"Problemas en la prediccion temperatura. {e}", status=500, mimetype='text/plain')            
        
        try:
            device = secrets.device
            path_hr_checkpoints = secrets.path_hr_checkpoints
            path_hr_model = secrets.path_hr_model
            model = torch.load(Path(path_hr_model) , map_location='cpu')
            model.to(device)
            
            trainer = tr_pmodel.TorchTrainer(model=model,
                                            device=device,
                                            checkpoint_folder= path_hr_checkpoints)
            trainer._load_best_checkpoint()
            y_pred_hr = trainer.predict_one(torch.unsqueeze(y_pred_zmodel[...,1], dim=-1))
            y_pred_hr = torch.squeeze(y_pred_hr)
            print(f"{y_pred_hr.shape=}")
        except Exception as e:
            # logger.info(f"[Prediccion] No es posible conectar con Cesens. {e}")
            return Response(f"Problemas en la prediccion hr. {e}", status=500, mimetype='text/plain')    

        try:
            device = secrets.device
            path_precipitacion_checkpoints = secrets.path_precipitacion_checkpoints
            path_precipitacion_model = secrets.path_precipitacion_model
            model = torch.load(Path(path_precipitacion_model) , map_location='cpu')
            model.to(device)
            
            trainer = tr_pmodel.TorchTrainer(model=model,
                                            device=device,
                                            checkpoint_folder= path_precipitacion_checkpoints)
            trainer._load_best_checkpoint()
            y_pred_precipitacion = trainer.predict_one(y_pred_zmodel[...,2:])
            y_pred_precipitacion = torch.squeeze(y_pred_precipitacion)
            print(f"{y_pred_precipitacion.shape=}")
        except Exception as e:
            # logger.info(f"[Prediccion] No es posible conectar con Cesens. {e}")
            return Response(f"Problemas en la prediccion precipitacion. {e}", status=500, mimetype='text/plain') 
        
        ## DESESCALADO    
        try:
            y_t = y_pred_temp.cpu().numpy()
            scaler = Escalador(Xmax=secrets.escaladores['temperatura']['max'],
                               Xmin=secrets.escaladores['temperatura']['min'],
                               min=0, max=1, auto_scale=False)
            y_t = scaler.inverse_transform(y_t)
            
            y_hr = y_pred_hr.cpu().numpy()
            scaler = Escalador(Xmax=secrets.escaladores['hr']['max'],
                               Xmin=secrets.escaladores['hr']['min'],
                               min=0, max=1, auto_scale=False)
            y_hr = scaler.inverse_transform(y_hr)
            
            y_rain = y_pred_precipitacion.cpu().numpy()
            bins = np.array(secrets.bins)
            y_rain = bins[np.argmax(y_rain, axis=-1)]
            scaler = Escalador(Xmax=secrets.escaladores['precipitacion']['max'],
                               Xmin=secrets.escaladores['precipitacion']['min'],
                               min=0, max=1, auto_scale=False)
            y_rain = scaler.inverse_transform(y_rain)
            
        except Exception as e:
            # logger.info(f"[Prediccion] No es posible conectar con Cesens. {e}")
            return Response(f"Problemas en el desescalado. {e}", status=500, mimetype='text/plain')             
            
        return make_response(jsonify(dict({"timestamp": now.timestamp(),
                                           "estacion": secrets.estacion,
                                           "data": [
                                               {"temp": y_t.tolist()},
                                               {"hr": y_hr.tolist()},
                                               {"rain": y_rain.tolist()}
                                               ]})), 200)
