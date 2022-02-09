import datetime
import logging
from flask_restful import Resource
from flask import jsonify, make_response, Response
from api.env import secrets
from api import api_modules

import torch
import numpy as np
from pathlib import Path
from common.utils.trainers import trainerSeq2Seq as tr
from common.utils.trainers import trainerpmodel as tr_pmodel
from common.utils.scalers import Escalador


logger = logging.getLogger(__name__)

class Prediccion(Resource):
       
    def get(self):
        token = "Token " + secrets.token_cesens
        now = datetime.datetime.now()
        past = now - datetime.timedelta(days=45) 
        
        ### PASADO
        try:
            dfs = api_modules.fetch_pasado(url=secrets.url_cesens, 
                                                 token=token, 
                                                 estaciones=secrets.estaciones_zona,
                                                 metricas=secrets.estaciones_metricas,
                                                 now=now,
                                                 past=past)
            logger.info("[Prediccion] Datos obtenidos de Cesens")
            df_pasado  = api_modules.generar_variables_pasado(estaciones=dfs, 
                                                              outliers=secrets.outliers_zona,
                                                              pasado=secrets.pasado,
                                                              now=now,
                                                              escaladores=secrets.escaladores,
                                                              cdg=secrets.CdG)
            logger.info("[Prediccion] Datos de Cesens convertidos")
        except Exception as e:
            logger.error(f"[Prediccion] No es posible conectar con Cesens. {e}")
            return Response(f"Problemas en la conexion con Cesens. {e}", status=500, mimetype='text/plain') 
        ### FUTURO
        try:
            df_futuro = api_modules.fetch_futuro(url=secrets.url_nwp, 
                                                 now=now.replace(minute=0, hour=0, second=0, microsecond=0),
                                                 future=72)
            logger.info("[Prediccion] Datos obtenidos del proveedor de predicciones")
            nwp = api_modules.generar_variables_futuro(df_futuro,
                                                       estaciones=df_pasado,
                                                       escaladores=secrets.escaladores)
            logger.info("[Prediccion] Datos del proveedor de predicciones convertidos")
        except Exception as e:
            logger.error(f"[Prediccion] Problemas en la conexion con NWP. {e}")
            return Response(f"Problemas en la conexion con NWP. {e}", status=500, mimetype='text/plain') 
        ## PREDICCION ZONAL
        device = secrets.device
        logger.info(f"[Prediccion] Usando {device} como dispositivo de calculo")
        try:
            Xf, Yt, P = api_modules.cargador_datos(datasets=df_pasado,nwps=nwp, pasado=secrets.pasado, futuro=72,
                                                   etiquetaF=secrets.Ff, etiquetaT=secrets.Ft, etiquetaP=secrets.Fnwp)
            
            path_zmodel_checkpoints = secrets.path_zmodel_checkpoints
            path_zmodel = secrets.path_zmodel
            model = torch.load(Path(path_zmodel) , map_location='cpu')
            model.to(device)
            trainer = tr.TorchTrainer(model=model,
                                    device=device,
                                    checkpoint_folder= path_zmodel_checkpoints)
            trainer._load_best_checkpoint()
            logger.info("[Prediccion] Cargado modelo zmodel")
            y_pred = trainer.predict_one(Xf, Yt, P)
            logger.info(f"[Prediccion] Prediccion obtenida con shape: {y_pred.shape}")
            
            y_t = torch.mean(y_pred[..., 0], dim=0)
            y_hr = torch.mean(y_pred[..., 1], dim=0)
            y_rain = torch.mean(y_pred[..., 2:], dim=0)
            y_t = torch.unsqueeze(y_t, dim=-1)
            y_hr = torch.unsqueeze(y_hr, dim=-1)
            y_pred_zmodel = torch.cat([y_t,y_hr,y_rain], dim=-1)
            y_pred_zmodel = torch.unsqueeze(y_pred_zmodel, dim=0)

        except Exception as e:
            logger.info(f"Problemas en la prediccion zonal. {e}")
            return Response(f"Problemas en la prediccion zonal. {e}", status=500, mimetype='text/plain') 
            
        ## MODELOS ESPECIFICOS DE LA PARCELA
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
            logger.info("[Prediccion] Cargado modelo de temperatura")
            y_pred_temp = trainer.predict_one(torch.unsqueeze(y_pred_zmodel[...,0], dim=-1))
            y_pred_temp = torch.squeeze(y_pred_temp)
        except Exception as e:
            logger.error(f"Problemas en la prediccion temperatura. {e}")
            return Response(f"Problemas en la prediccion temperatura. {e}", status=500, mimetype='text/plain')            
        
        try:
            device = secrets.device
            path_hr_checkpoints = secrets.path_hr_checkpoints
            path_hr_model = secrets.path_hr_model
            model = torch.load(Path(path_hr_model) , map_location='cpu')
            model.to(device)
            logger.info("[Prediccion] Cargado modelo de HR")
            trainer = tr_pmodel.TorchTrainer(model=model,
                                            device=device,
                                            checkpoint_folder= path_hr_checkpoints)
            trainer._load_best_checkpoint()
            y_pred_hr = trainer.predict_one(torch.unsqueeze(y_pred_zmodel[...,1], dim=-1))
            y_pred_hr = torch.squeeze(y_pred_hr)
        except Exception as e:
            logger.error(f"[Prediccion] Problemas en la prediccion HR. {e}")
            return Response(f"Problemas en la prediccion HR. {e}", status=500, mimetype='text/plain')    

        try:
            device = secrets.device
            path_precipitacion_checkpoints = secrets.path_precipitacion_checkpoints
            path_precipitacion_model = secrets.path_precipitacion_model
            model = torch.load(Path(path_precipitacion_model) , map_location='cpu')
            model.to(device)
            logger.info("[Prediccion] Cargado modelo de Lluvia")
            trainer = tr_pmodel.TorchTrainer(model=model,
                                            device=device,
                                            checkpoint_folder= path_precipitacion_checkpoints)
            trainer._load_best_checkpoint()
            y_pred_precipitacion = trainer.predict_one(y_pred_zmodel[...,2:])
            y_pred_precipitacion = torch.squeeze(y_pred_precipitacion)
            print(f"{y_pred_precipitacion.shape=}")
        except Exception as e:
            logger.error(f"[Prediccion] Problemas en la prediccion LLuvia {e}")
            return Response(f"Problemas en la prediccion LLuvia. {e}", status=500, mimetype='text/plain') 
        
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
            logger.info("[Prediccion] Desescalado completado")
        except Exception as e:
            logger.error(f"[Prediccion] Problemas en el desescalado. {e}")
            return Response(f"Problemas en el desescalado. {e}", status=500, mimetype='text/plain')             
        logger.info("[Prediccion] Generando JSON de salida")    
        return make_response(jsonify(dict({"timestamp": now.timestamp(),
                                           "estacion": secrets.estacion,
                                           "data": [
                                               {"temp": y_t.tolist()},
                                               {"hr": y_hr.tolist()},
                                               {"rain": y_rain.tolist()}
                                               ]})), 200)
