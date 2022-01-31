"""
Diversas funciones para su uso dentro de la generacion de datasets tanto para zmodel y pmodel

Comun:
- distanciasCdG(latitudes: list, longitudes: list, altitudes: list) -> tuple
- haversine(lat1: float, long1: float, lat2: float, long2: float) -> float
- parser_experiment(cfg: dict, name: str) -> dict
- temporales(column_date: pd.Series) -> tuple

Zmodel:
- generarvariablesZmodel(estaciones: list=None, escaladores: list=None, outliers: list=None) -> tuple

Pmodel:
- extraccionFuturo(df: pd.DataFrame, k: int=30 ) -> np.array
"""


from shutil import ExecError
import pandas as pd
import numpy as np
import yaml
import pickle
from pathlib import Path
from tqdm import tqdm
from colorama import Fore, Back, Style
from math import sqrt, sin, cos, atan2, pi
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import common.predict.modules as predictor
import common.utils.parser as parser
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder

from datetime import datetime, timedelta
import common.utils.indicesbioclimaticos as bio
from common.utils.otrosindicadores import macd
from common.utils.scalers import Escalador
from attrdict import AttrDict
from common.utils import errors

OK = "\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]"
FAIL = "\t[ " + Fore.RED + "FAIL" + Style.RESET_ALL + " ]"

def no_NaNs(df: pd.DataFrame=None) -> bool:
    """ Chequea si un dataframe tiene NaNs"""
    
    return False if df.isna().sum().sum() > 0 else True


def distanciasCdG(latitudes: list, longitudes: list, altitudes: list) -> tuple:
    """Calcula el centro de gravedad de una lista de coordenadas

    Args:
        latitudes (list): Lista de latitudes
        longitudes (list): Lista de longitudes
        altitudes (list): Lista de altitudes

    Returns:
        tuple: (latitud del cdg, longitud del cdg, altitud del cdg)
    """
    return np.array(latitudes).mean(), np.array(longitudes).mean(), np.array(altitudes).mean()
    
def haversine(lat1: float, long1: float, lat2: float, long2: float) -> float:
    """Calcula la distancia entre dos punto dados por latitud y longitud

    Args:
        lat1 (float): Latitud punto 1
        long1 (float): Longitud punto 1
        lat2 (float): Latitud punto 2
        long2 (float): Longitud punto 2

    Returns:
        float: Distancia en km
    """
    degree_to_rad = float(pi / 180.0)
    d_lat = (lat2 - lat1) * degree_to_rad
    d_long = (long2 - long1) * degree_to_rad
    a = pow(sin(d_lat / 2), 2) + cos(lat1 * degree_to_rad) * cos(lat2 * degree_to_rad) * pow(sin(d_long / 2), 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    km = 6367 * c
    return abs(km)

def temporales(df: pd.DataFrame= None, trig: bool=False) -> pd.DataFrame:
    if 'fecha' not in df.columns:
        df['fecha'] = df.index
        borrar_fecha = True
        
    df['dia'] = df['fecha'].apply(lambda x: x.day / 31)
    df['mes'] = df['fecha'].apply(lambda x: x.month / 12)
    df['hora'] = df['fecha'].apply(lambda x: x.hour / 24)
    if borrar_fecha: df.drop(columns=['fecha'], inplace=True)
    return df

def check_variables_entrada(df: pd.DataFrame=None) -> bool:
    """Comprueba que el dataset contenga columnas de temperatura, hr y precipitacion"""
    
    return False if 'temperatura' not in df.columns or 'hr' not in df.columns or 'precipitacion' not in df.columns else True

def check_outliers(df: pd.DataFrame=None, metricas: list=None, outliers: dict=None) -> bool:
    """Comprueba que no haya outliers en el dataset"""

    for metrica in metricas:
        outlier = outliers[metrica]
        if any(df[metrica] > outlier['max'] * (1 + outlier['tol'] / 100)):
            df.loc[df[metrica] > outlier['max'] * (1 + outlier['tol'] / 100), metrica] = outlier['max']
        if any(df[metrica] < outlier['min'] * (1 + outlier['tol'] / 100)):
            df.loc[df[metrica] < outlier['min'] * (1 + outlier['tol'] / 100), metrica] = outlier['min']
    return df

def check_longitud(df: pd.DataFrame=None, longitud: int=None) -> bool:
    """Comprueba la longitud de un dataset"""
    
    return True if len(df) > longitud else False
    
def quitar_nans(df: pd.DataFrame=None) -> pd.DataFrame:
    df = df.fillna(df.mean())
    return df

def variables_bioclimaticas(df: pd.DataFrame=None) -> tuple:
    var_bio =[]
    df['integraltermica'] = bio.integral_termica(temperatura=np.array(df['temperatura'].values))
    var_bio.append('integraltermica')
    df['integralrain'] = bio.integral_precipitacion(arr=np.array(df['precipitacion'].values,), periodo=3 * 30)
    df['indiceangtrom'] = bio.indice_angtrom(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
    var_bio.append('indiceangtrom')
    df['indicemartonne'] = bio.indice_martonne(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values),muestras_dia=24)
    var_bio.append('indicemartonne')
    df['indicelang'] = bio.indice_lang(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
    var_bio.append('indicelang')
    return (var_bio, df)        
    
def variables_macd(df: pd.DataFrame=None) -> tuple:
    var_macd = []
    df['tempmacd'] = macd(np.array(df['temperatura'].values))
    var_macd.append('tempmacd')
    df['hrmacd'] = macd(np.array(df['hr'].values))
    var_macd.append('hrmacd')
    df['precipitacionmacd'] = macd(np.array(df['precipitacion'].values))
    var_macd.append('precipitacionmacd')
    return (var_macd, df)         

def recortar(df: pd.DataFrame, fecha_maxima: datetime, pasado: int):
    df = df.loc[df.index <= fecha_maxima, :]
    return df.iloc[len(df) - pasado:len(df), :]
    
        
def generar_variables(estaciones: list=None, outliers: dict=None, pasado: int=None, now: datetime=None, predicciones: dict=None, CdG: tuple=None) -> tuple:
    
    fechas_maximas = []
    for i, estacion in enumerate(estaciones):
    
        if estacion['fecha_maxima'] - estacion['fecha_minima'] < timedelta(days=366):
            print(f"[API][generar_variables] OMIT Estacion: {estacion['nombre']} No tiene mas de un año de pasado")
            estaciones.pop(i)
            continue
        if now - estacion['fecha_maxima'] > timedelta(hours=2):
            print(f"[API][generar_variables] OMIT Estacion: {estacion['nombre']} Su ultimo envio es posterior a 2 horas")
            estaciones.pop(i)
            continue
        if len(estacion[1]) != len(estacion[6]) or len(estacion[11]) != len(estacion[6]):
            print(f"[API][generar_variables] OMIT Estacion: {estacion['nombre']} Las medidas de la estacion no estan sincronizadas")
            estaciones.pop(i)
            continue
        fechas_maximas.append(estacion['fecha_maxima'])
        
    fecha_maxima = min(fechas_maximas)
        
    dfs, altitudes, longitudes, latitudes = [], [], [], []
    var_bio, var_macd = [], []
    
    for estacion in estaciones:
        df = {"fecha": estacion['fecha'], "temperatura": estacion[1], "hr": estacion[6], "precipitacion": estacion[11]}
        df = pd.DataFrame(df)
        df['fecha'] = pd.to_datetime(df['fecha'], format="%Y-%m-%d %H:%M:%S")
        df.set_index('fecha', drop=True, inplace=True)

        df = check_outliers(df, metricas=["temperatura", "hr", "temperatura"], outliers=outliers)
        df = df.resample('1H').interpolate()
        
        if df.isna().sum().sum() > 0:
            print(f"[API][generar_variables] WARN Estacion: {estacion['nombre']} Hay Nans! en los datos")
            df = quitar_nans(df)
       
        var_bio, df = variables_bioclimaticas(df)
        if df.isna().sum().sum() > 0:
            print(f"[API][generar_variables] WARN Estacion: {estacion['nombre']} Hay Nans! en los bioclimaticos")
            df = quitar_nans(df)

        var_macd, df = variables_macd(df)
        if df.isna().sum().sum() > 0:
            print(f"[API][generar_variables] WARN Estacion: {estacion['nombre']} Hay Nans! en macd")
            df = quitar_nans(df)
            
        df = recortar(df, fecha_maxima, pasado)

        df = temporales(df)
        if df.isna().sum().sum() > 0:
            print(f"[API][generar_variables] WARN Estacion: {estacion['nombre']} Hay Nans! en temporales")
            df = quitar_nans(df)

        longitudes.append(estacion["longitud"])
        latitudes.append(estacion["latitud"])
        altitudes.append(estacion["altitud"])
                          
        dfs.append(df)

    cdg = distanciasCdG(latitudes, longitudes, altitudes)
    for i, df in enumerate(dfs):
        df['distancia'] = haversine(lat1=latitudes[i], long1=longitudes[i], lat2=cdg[0], long2=cdg[1])
        df['altitud'] = altitudes[i] - cdg[2]
        df.reset_index(drop=True)

    parametros = dict()
    for metrica in (['temperatura', 'hr', 'precipitacion'] + var_bio + var_macd):
        parametros[metrica] = dict()
        parametros[metrica]['max'] = float(max([_[metrica].max() for _ in dfs]))
        parametros[metrica]['min'] = float(min([_[metrica].min() for _ in dfs]))
        scaler = Escalador(Xmax=parametros[metrica]['max'], Xmin=parametros[metrica]['min'], min=0, max=1, auto_scale=False)
        for df in dfs:
            df[metrica] = scaler.transform(np.array(df[metrica].values))
            if not no_NaNs(df):
                print(f"[API][generar_variables] FAIL Estacion: {estacion} Hay Nans! en el escalador")
                raise Exception(f"[API][generar_variables] FAIL Estacion: {estacion} Hay Nans! en el escalador")

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
            print(f"[API][generar_variables] FAIL Estacion: {estacion} Hay Nans! en el escalador")
            raise Exception(f"[API][generar_variables] FAIL Estacion: {estacion} Hay Nans! en el escalador")
                         
    metadata = {}
    metadata["longitud"] = len(dfs)
    metadata["datasets"] = {}
    metadata["CdG"] = [float(_) for _ in list(cdg)]
    metadata['fecha_max'] = datetime.strftime(fecha_maxima, format="%Y-%m-%d %H:%M:%S")
    metadata['indice_min'] = min([_.index.min() for _ in dfs])
    metadata['indice_max'] = max([_.index.max() for _ in dfs])
    metadata['escaladores'] = {}
    metadata['escaladores'] = parametros
    metadata['bins'] = bins
    return (dfs, metadata)