import logging
import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

from datetime import datetime, timedelta

from common.utils.scalers import Escalador

from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re

import requests
import json
from typing import List

import numpy as np
import pandas as pd
import torch
from common.data_preprocessing.modules import (no_NaNs,
                                               haversine,
                                               temporales,
                                               check_outliers,
                                               quitar_nans,
                                               variables_bioclimaticas,
                                               variables_macd
                                               )

logger = logging.getLogger(__name__)



def quitar_outliers(df: pd.DataFrame=None, metricas: list=None, outliers: dict=None) -> bool:
    """Comprueba que no haya outliers en el dataset"""

    for metrica in metricas:
        outlier = outliers[metrica]
        if any(df[metrica] > outlier['max'] * (1 + outlier['tol'] / 100)):
            df.loc[df[metrica] > outlier['max'] * (1 + outlier['tol'] / 100), metrica] = outlier['max']
        if any(df[metrica] < outlier['min'] * (1 + outlier['tol'] / 100)):
            df.loc[df[metrica] < outlier['min'] * (1 + outlier['tol'] / 100), metrica] = outlier['min']
    return df


def recortar(df: pd.DataFrame, fecha_maxima: datetime, pasado: int):
    df = df.loc[df.index <= fecha_maxima, :]
    return df.iloc[len(df) - pasado:len(df), :]
    
def generar_variables_pasado(estaciones: list, outliers: dict, pasado: int, now: datetime, escaladores: dict, cdg: list) -> List[pd.DataFrame]:
   
    estaciones_def = []
    fechas_maximas = []
    for i, estacion in enumerate(estaciones):
    
        if estacion['fecha_maxima'] - estacion['fecha_minima'] < timedelta(days=30):
            logger.info(f"[API][generar_variables] OMIT Estacion: {estacion['nombre']} No tiene mas de un año de pasado")
            continue
        elif now - estacion['fecha_maxima'] > timedelta(hours=2):
            logger.info(f"[API][generar_variables] OMIT Estacion: {estacion['nombre']} Su ultimo envio es posterior a 2 horas")
            continue
        elif len(estacion[1]) != len(estacion[6]) or len(estacion[11]) != len(estacion[6]):
            logger.info(f"[API][generar_variables] OMIT Estacion: {estacion['nombre']} Las medidas de la estacion no estan sincronizadas")
            continue
        elif len(set(estacion['fechas_parciales']))!=1:  # No hay sincronismo
            logger.info(f"[API][generar_variables] OMIT Estacion: {estacion['nombre']} No hay sincronismo. Fechas máximas no coinciden")
            continue
        else:
            estaciones_def.append(estacion)
            fechas_maximas.append(estacion['fecha_maxima'])
    
    estaciones = estaciones_def
    fecha_maxima = min(fechas_maximas)

    dfs, altitudes, longitudes, latitudes = [], [], [], []
    var_bio, var_macd = [], []
    
    for estacion in estaciones:
        df = {"fecha": estacion['fecha'], "temperatura": estacion[1], "hr": estacion[6], "precipitacion": estacion[11]}
        df = pd.DataFrame(df)
        df['fecha'] = pd.to_datetime(df['fecha'], format="%Y-%m-%d %H:%M:%S")
        df.set_index('fecha', drop=True, inplace=True)
        if not check_outliers(df, metricas=["temperatura", "hr", "precipitacion"], outliers=outliers):
            logger.info(f"[API][generar_variables] Estacion: {estacion['nombre']} Hay outliers... eliminacion automatica")
            df = quitar_outliers(df, metricas=["temperatura", "hr", "precipitacion"], outliers=outliers)
            if df.isna().sum().sum() > 0:
                logger.info(f"[API][generar_variables] Estacion: {estacion['nombre']} Hay Nans! despues de quitar outliers")
                raise Exception(f"[API][generar_variables] Estacion: {estacion['nombre']} Hay Nans! despues de quitar outliers")
        df = df.resample('1H').interpolate()

        if df.isna().sum().sum() > 0:
            logger.info(f"[API][generar_variables] WARN Estacion: {estacion['nombre']} Hay Nans! en los datos")
            df = quitar_nans(df)
       
        var_bio, df = variables_bioclimaticas(df)
        if df.isna().sum().sum() > 0:
            print(f"[API][generar_variables] WARN Estacion: {estacion['nombre']} Hay Nans! en los bioclimaticos")
            df = quitar_nans(df)

        var_macd, df = variables_macd(df)
        if df.isna().sum().sum() > 0:
            logger.info(f"[API][generar_variables] WARN Estacion: {estacion['nombre']} Hay Nans! en macd")
            df = quitar_nans(df)
            
        df = recortar(df, fecha_maxima, pasado)

        df = temporales(df)
        if df.isna().sum().sum() > 0:
            print(f"[API][generar_variables] WARN Estacion: {estacion['nombre']} Hay Nans! en temporales")
            df = quitar_nans(df)

        longitudes.append(estacion["longitud"])
        latitudes.append(estacion["latitud"])
        altitudes.append(estacion["altitud"])
                          
        dfs.append(df.copy())


    for i, df in enumerate(dfs):
        df['distancia'] = haversine(lat1=latitudes[i], long1=longitudes[i], lat2=cdg[0], long2=cdg[1])
        df['altitud'] = altitudes[i] - cdg[2]
        df.reset_index(drop=True, inplace=True)

    for metrica in (['temperatura', 'hr', 'precipitacion'] + var_bio + var_macd):
        scaler = Escalador(Xmax=escaladores[metrica]['max'], Xmin=escaladores[metrica]['min'], min=0, max=1, auto_scale=False)
        for df in dfs:
            df[metrica] = scaler.transform(np.array(df[metrica].values))
            if not no_NaNs(df):
                logger.error(f"[API][generar_variables] FAIL Estacion:  Hay Nans! en el escalador")
                raise Exception(f"[API][generar_variables] FAIL Estacion:  Hay Nans! en el escalador")

    bins = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 1.0]
    ohe = OneHotEncoder(dtype=int, sparse=False)
    # Necesitamos que todas las clases esten representadas para ello debo fit el transformer con todas
    ohe.fit(np.arange(len(bins)).reshape(-1,1))
    for df in dfs:
        df['clase_precipitacion'] = df['precipitacion'].apply(lambda x: np.digitize(x, bins) - 1)
        df_aux1 = pd.DataFrame(ohe.transform(df['clase_precipitacion'].values.reshape(-1,1)),
                               columns=["clase_precipitacion_" + str(_) for _ in range(len(bins))])
        df.drop(columns=['clase_precipitacion'], inplace=True)
        for _ in range(len(bins)):
            df['clase_precipitacion_' + str(_)] = df_aux1['clase_precipitacion_' + str(_)].values
        
        if not no_NaNs(df):
            print(f"[API][generar_variables] FAIL Estacion:  Hay Nans! en el clasificador")
            raise Exception(f"[API][generar_variables] FAIL Estacion: Hay Nans! en el clasificador")
    return dfs

def fetch_futuro(url: str, now: datetime, future: int ) -> pd.DataFrame:
    futuro = {"fecha": [], "temperatura": [], "hr": [], "precipitacion": []}
    _ = Request("https://www.eltiempo.es/villaverde-de-rioja.html?v=por_hora")  #para que se cachee
    req = Request(url)
    html_page = urlopen(req)
    html_text = BeautifulSoup(html_page, "html.parser").get_text()
    result = re.sub(r"(\n){1,10}", r"\n", html_text)
    result = re.sub(r"(\t){1,10}", " ", result)
    
    horas = re.findall(r"\n[0-9]{1,2}:[0-9]{1,2}\n", result)
    dia=0
    cuenta = 0
    for hora in horas:
        hora = re.sub(r"\n","", hora)
        hora = int(re.split(r":", hora)[0])
        if hora == 0:
            dia = dia + 1
        futuro['fecha'].append((now + timedelta(hours=hora, days=dia)))
    cuenta = future + 1 - len(futuro['fecha'])
    for _ in range(cuenta): # 71 a 73
        futuro['fecha'].append((now + timedelta(hours=hora + 1, days=dia)))
    
    temperaturas = re.findall(r"00\n-?[0-9]+°", result)
    for temperatura in temperaturas:
        t = re.sub(r"°","", temperatura)
        t = int(re.split(r"\n", t)[1])
        futuro['temperatura'].append(int(t))
    for _ in range(cuenta):
        futuro['temperatura'].append(int(t))      
    
    humedades = re.findall(r"\nHumedad\n[0-9]+%\n", result)
    for humedad in humedades:
        h = re.sub(r"%", "", humedad)
        h = re.split(r"\n", h)[2]
        futuro['hr'].append(int(h))
    for _ in range(cuenta):
        futuro['hr'].append(int(h))
    
    precipitaciones = re.findall(r"\nLluvias\n[0-9]+.?[0-9]? mm\n", result)
    for precipitacion in precipitaciones:
        h = re.sub(r" mm", "", precipitacion)
        h = re.split(r"\n", h)[2]
        futuro['precipitacion'].append(float(h))
    for _ in range(cuenta):
        futuro['precipitacion'].append(float(h))
    
    return pd.DataFrame(futuro)

def fetch_pasado(url: str, token: str, estaciones: list, metricas: list, now: datetime, past:datetime) -> List[pd.DataFrame]:
    dfs = []
    
    for estacion in estaciones: 
        id, datos = list(estacion.items())[0]
        data = {}
        data['altitud'] = datos['altitud']
        data['longitud'] = datos['longitud']
        data['latitud'] = datos['latitud']
        data['nombre'] = datos['nombre']
        fechas_parciales=[]
        try:
            for metrica in metricas:
                api_string = f"{url}/api/datos/{str(id)}/{str(metrica)}/{past.strftime('%Y%m%d')}-{now.strftime('%Y%m%d')}"
                data_response_cesens = requests.get(api_string, headers={"Content-Type": "application/json", "Authentication": f"{token}"})
                data[metrica] = [v for k, v in dict(json.loads(data_response_cesens.text)).items()]
                fechas_parciales.append(max([datetime.fromtimestamp(int(k)) for k, v in dict(json.loads(data_response_cesens.text)).items()]))
        except:
            logger.info(f"[API][fetch_pasado] FAIL Estacion: {data['nombre']}")
            continue

        data['fechas_parciales'] = fechas_parciales
        data['fecha'] = [datetime.fromtimestamp(int(k))
                            for k, _ in dict(json.loads(data_response_cesens.text)).items()] ## str -> timestamp -> str
        data['fecha_maxima'] = max(data['fecha'])
        data['fecha_minima'] = min(data['fecha'])
        
        
        dfs.append(data)
    return dfs

def generar_variables_futuro(nwp: pd.DataFrame, escaladores: dict, estaciones: list ) -> List[pd.DataFrame]:
    
    dfs = []
    nwp.set_index('fecha', drop=True, inplace=True)
    nwp = temporales(nwp)
    
    for metrica in ['temperatura', 'hr', 'precipitacion']:
        scaler = Escalador(Xmax=escaladores[metrica]['max'], Xmin=escaladores[metrica]['min'], min=0, max=1, auto_scale=False)
        nwp[metrica] = scaler.transform(np.array(nwp[metrica].values))
        if not no_NaNs(nwp):
            logger.error(f"[API][generar_variables_futuro] FAIL Estacion: NWP Hay Nans! en el escalador")
            raise Exception(f"[API][generar_variables_futuro] FAIL Estacion: NWP Hay Nans! en el escalador")
            
    nwp.rename(columns = {'temperatura': 'nwp_temperatura'}, inplace = True)
    nwp.rename(columns = {'hr':'nwp_hr'}, inplace = True)
    nwp.rename(columns = {'precipitacion':'nwp_precipitacion'}, inplace = True)
    
    bins = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 1.0]
    ohe = OneHotEncoder(dtype=int, sparse=False)
    # Necesitamos que todas las clases esten representadas para ello debo fit el transformer con todas
    ohe.fit(np.arange(len(bins)).reshape(-1,1))
    nwp['nwp_clase_precipitacion'] = nwp['nwp_precipitacion'].apply(lambda x: np.digitize(x, bins) - 1)
    df_aux1 = pd.DataFrame(ohe.transform(nwp['nwp_clase_precipitacion'].values.reshape(-1,1)),
                           columns=["nwp_clase_precipitacion_" + str(_) for _ in range(len(bins))])
    nwp.drop(columns=['nwp_clase_precipitacion'], inplace=True)
    for _ in range(len(bins)):
        nwp['nwp_clase_precipitacion_' + str(_)] = df_aux1['nwp_clase_precipitacion_' + str(_)].values
    if not no_NaNs(nwp):
        logger.error(f"[API][generar_variables_futuro] FAIL Estacion: NWP Hay Nans! en el clasificador")
        raise Exception(f"[API][generar_variables] FAIL Estacion: NWP Hay Nans! en el clasificador")
    
    nwp.reset_index(drop=True, inplace=True)
    
    for estacion in estaciones:
        distancia = estacion.loc[0, 'distancia']
        altitud = estacion.loc[0, 'altitud']
        nwp['distancia'] = distancia
        nwp['altitud'] = altitud
        dfs.append(nwp.copy())    
    
    return dfs

def cargador_datos(datasets: list, nwps: pd.DataFrame, pasado: int, futuro: int, etiquetaF: list=None, etiquetaT: list=None, etiquetaP: list=None) -> tuple:
        
    batches = len(datasets)
    X_f = np.empty(shape=(batches, # batches
                              pasado, # sequences
                              len(etiquetaF)))  # features 
    Y_t = np.empty(shape=(batches, # batches
                              futuro + 1, # sequences
                              len(etiquetaT)))  # etiquetaF
    P = np.empty(shape=(batches, # batches
                            futuro + 1, # sequences + 1 
                            len(etiquetaP)))  # etiquetaP
        
    for i, (df, nwp) in enumerate(zip(datasets, nwps)):
        X_f[i] = df.loc[: , etiquetaF].values
        Y_t[i] = nwp.loc[:, etiquetaT].values
        P[i] = nwp.loc[:, etiquetaP].values
    
    return (torch.from_numpy(X_f).float(),  # (batches, Lx + 1, Ff)
                torch.from_numpy(Y_t).float(),  # (batches, Ly + 1, Ft)
                torch.from_numpy(P).float())  # (batches, Ly + 1, Fnwp)
    
