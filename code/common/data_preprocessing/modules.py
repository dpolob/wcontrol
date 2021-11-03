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

from datetime import datetime, timedelta
import common.utils.indicesbioclimaticos as bio
from common.utils.otrosindicadores import macd
from common.utils.scalers import Escalador
from attrdict import AttrDict

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
    """Generador de variables relativas a la fecha

    Args:
        df: (pd.DataFrame): Dataset sobre al que insertar las nuevas columnas (se hace asi para facilidad de cambio)
        trig: (bool): Uso de variables codificadas como seno y coseno o variables escaladas

    Returns:
        pd.DataFrame actualizado con (día_sin, dia_cos, mes_sin, mes_cos, hora_sin, hora_cos) si trig == True 
        o actualizado con (día, mes, hora) si trig == False
    """ 
    if 'fecha' not in df.columns:
        df['fecha'] = df.index
        borrar_fecha = True
        
    if trig:    
        df['dia'] = df['fecha'].apply(lambda x: x.day * 2 * pi / 31)
        df['dia_sin'] = np.sin(df['dia'])
        df['dia_cos'] = np.cos(df['dia'])
        
        df['mes'] = df['fecha'].apply(lambda x: x.month * 2 * pi  / 12)
        df['mes_sin'] = np.sin(df['mes'])
        df['mes_cos'] = np.cos(df['mes'])
        
        df['hora'] = df['fecha'].apply(lambda x: x.hour * 2 * pi  / 24)
        df['hora_sin'] = np.sin(df['hora'])
        df['hora_cos'] = np.cos(df['hora'])
        
        df.drop(columns=['dia', 'mes', 'hora'], inplace=True)
        if borrar_fecha: df.drop(columns=['fecha'], inplace=True)
    else:
        df['dia'] = df['fecha'].apply(lambda x: x.day / 31)
        df['mes'] = df['fecha'].apply(lambda x: x.month / 12)
        df['hora'] = df['fecha'].apply(lambda x: x.hour / 24)
        if borrar_fecha: df.drop(columns=['fecha'], inplace=True)
    return df


def check_variables_entrada(df: pd.DataFrame=None) -> bool:
    if 'temperatura' not in df.columns or 'hr' not in df.columns or 'precipitacion' not in df.columns:
        tqdm.write(FAIL)
        return False
    else:
        tqdm.write(OK)
        return True

def check_outliers(df: pd.DataFrame=None, metricas: list=None, outliers: list=None) -> bool:
    for metrica in metricas:
        outlier = outliers[metrica]
        if metrica in df.columns:
            if any(df[metrica] > outlier['max'] * (1 + outlier['tol'] / 100)):
                tqdm.write(f"Valores de {metrica} superan el valor de {outlier['max']}" + FAIL)
                return False
            if any(df[metrica] < outlier['min'] * (1 + outlier['tol'] / 100)):
                tqdm.write(f"Valores de {metrica} por debajo el valor de {outlier['min']}" + FAIL)
                return False
    tqdm.write(OK)
    return True

def check_longitud(df: pd.DataFrame=None, longitud: int=None) -> bool:
    """Comprueba la longitud de un dataset"""
    
    if len(df) > longitud:
        tqdm.write(OK)
        return True
    else:
        tqdm.write(FAIL)
        return False
    
def quitar_nans(df: pd.DataFrame=None, how: str='mean') -> pd.DataFrame:
    """Elimina NaN segun el criterio dado por how

    Args:
        df (pd.DataFrame): dataframe al que quitar NaNs
        how (str, optional): Metodo de eliminacion:
            - IterativeImputer: Usando sklearn iterativeimputer, se usan las variables temporale 
            - mean: sustituye NaNs por la media
            Defaults to 'mean'.

    Returns:
        pd.DataFrame: dataframes sin NaN
    """
    tqdm.write(f"{df.isna().sum().sum()}", end='')
       
    if how == 'IterativeImputer':
        df = temporales(df)
        if not no_NaNs(df):
            imp_mean = IterativeImputer(random_state=0)
            idf=imp_mean.fit_transform(df)
            idf = pd.DataFrame(idf, columns=df.columns)
            df['temperatura'] = idf['temperatura'].values
            df['hr'] = idf['hr'].values
            df['precipitacion'] = idf['precipitacion'].values
            del idf
            if not no_NaNs(df):
                tqdm.write("Quedan NaNs en el dataset. Por favor compruebe")
                return None
    if how == 'mean': 
        df = df.fillna(df.mean()) 
    
    tqdm.write(" -> " + str(df.isna().sum().sum()), end='')
    tqdm.write("\t" + OK)
    return df

def variables_bioclimaticas(df: pd.DataFrame=None) -> tuple:
    """Genera las variables bioclimaticas

    Args:
        df (pd.DataFrame, optional): dataframe sobre el que calcular las variables.

    Returns:
        tuple: (nombres de las variables calculadas, dataframe con variables bioclimaticas)
    """
    
    var_bio =[]
    if set(['temperatura']).issubset(set(df.columns)): 
        df['integraltermica'] = bio.integral_termica(temperatura=np.array(df['temperatura'].values))
        var_bio.append('integraltermica')
    if set(['precipitacion']).issubset(set(df.columns)): 
        df['integralrain'] = bio.integral_precipitacion(arr=np.array(df['precipitacion'].values,), periodo=3 * 30)
        var_bio.append('integralrain')
    if set(['temperatura', 'precipitacion']).issubset(set(df.columns)): 
        df['indiceangtrom'] = bio.indice_angtrom(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
        var_bio.append('indiceangtrom')
        df['indicemartonne'] = bio.indice_martonne(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values),muestras_dia=24)
        var_bio.append('indicemartonne')
        df['indicelang'] = bio.indice_lang(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
        var_bio.append('indicelang')
    if not no_NaNs(df):
        tqdm.write("Se han generado NANs" + FAIL)
        return (None, None)
    else:
        tqdm.write(OK)
        return (var_bio, df)        
    
def variables_macd(df: pd.DataFrame=None) -> tuple:
    """Calcula las variables relacionadas con MACD

    Args:
        df (pd.DataFrame, optional): dataframe sobre el que calcular las variables.

    Returns:
        tuple: (nombres de las variables calculadas MACD, dataframe con variables MACD)
    """
    
    var_macd = []
    if set(['temperatura']).issubset(set(df.columns)):
        df['tempmacd'] = macd(np.array(df['temperatura'].values))
        var_macd.append('tempmacd')
    if set(['hr']).issubset(set(df.columns)):
        df['hrmacd'] = macd(np.array(df['hr'].values))
        var_macd.append('hrmacd')
    if set(['precipitacion']).issubset(set(df.columns)):
        df['precipitacionmacd'] = macd(np.array(df['precipitacion'].values))
        var_macd.append('precipitacionmacd')
    if not no_NaNs(df):
        tqdm.write("Se han generado NANs" + FAIL)
        return (None, None)
    else:
        tqdm.write(OK)
        return (var_macd, df)         

def extraccionFuturo(df: pd.DataFrame, k: int=30 ) -> np.array:
    data = np.empty(shape=(len(df) , k))  #ojo la longitud real es len(df) - k pero 
    # lo saco con la misma longitud que el df para que los pueda juntar los dataset
    for i in range(len(df) - k):
        for j in range(k):
            data[i, j] = df.iloc[i + j + 1, 0]
    return data
        
def generarvariablesZmodel(estaciones: list=None, outliers: list=None, proveedor: dict=None) -> tuple:
    """Wrapper para la generacion del dataset para el modelo zonal

    Args:
        estaciones (list, optional): lista de estaciones. Cada estacion es un diccionario.
        outliers (list, optional): lista de outliers
        proveedor (dict, optional): estacion con la prevision climatica para el futuro.

    Returns:
        tuple: (lista de dataset (uno por cada estacion), diccionario de metadatos del dataset)
    """
    
    dfs = []
    dfs_nombres = []
    for estacion in tqdm(estaciones, desc='Estaciones'):
        ruta = Path(estacion["ruta"])
        metricas = estacion["metricas"]
        
        tqdm.write(Fore.YELLOW + f"{estacion['nombre']}" + Style.RESET_ALL)
        tqdm.write(f"\tLeyendo estacion desde {ruta}", end='')
    
        if not ruta.is_file():
            tqdm.write(f"No existe el archivo para {estacion['nombre']}")
            tqdm.write(FAIL)
            continue
        else:
            df = pd.read_csv(ruta, sep=',', decimal='.', header='infer')
            df['fecha'] = pd.to_datetime(df['fecha'], format=estacion["format"] if "format" in estacion.keys() else "%Y-%m-%d %H:%M:%S")
            df.set_index('fecha', drop=True, inplace=True)
            tqdm.write(OK)
        
        tqdm.write(f"\tComprobando variables de entrada", end='')
        if not check_variables_entrada(df):
            continue
        
        tqdm.write(f"\tComprobando outliers", end='')
        if not check_outliers(df, metricas, outliers):
            exit()
        tqdm.write(OK)
        
        tqdm.write(f"\tResample a 1 hora", end='')
        df = df.resample('1H').interpolate()
        tqdm.write(OK)
        
        tqdm.write(f"\tComprobando longitud del dataset mayor de un año", end='')
        if not check_longitud(df, 365 * 24):
            continue
        
        tqdm.write(f"\tEliminando NaNs...", end='')
        df = quitar_nans(df)
        tqdm.write(OK)
       
        tqdm.write(f"\tAgregar predicciones", end='')
        nwp = pd.read_csv(proveedor.ruta, sep=',', decimal='.', header='infer')
        nwp['fecha'] = pd.to_datetime(nwp['fecha'], format=proveedor.format if "format" in proveedor.keys() else "%Y-%m-%d %H:%M:%S")
        nwp.rename(columns = {'temperatura': 'nwp_temperatura'}, inplace = True)
        nwp.rename(columns = {'hr':'nwp_hr'}, inplace = True)
        nwp.rename(columns = {'precipitacion':'nwp_precipitacion'}, inplace = True)
        nwp.set_index('fecha', drop=True, inplace=True)
        nwp = nwp.resample('1H').interpolate()
        nwp = quitar_nans(nwp)
        df = df.join(nwp, how='inner')
        nwp = quitar_nans(nwp)
        var_nwp = ['nwp_temperatura', 'nwp_hr', 'nwp_precipitacion']
        tqdm.write(OK)

        tqdm.write(f"\tCalculando variables climáticas", end='')
        var_bio, df = variables_bioclimaticas(df)
        if var_bio is None:
            exit()
          
        tqdm.write(f"\tCalculando MACD", end='')
        var_macd, df = variables_macd(df)
        if var_macd is None:
            exit()
       
        tqdm.write(f"\tCalculando variables cíclicas de fecha", end='')
        df = temporales(df)
        tqdm.write(OK)
        if no_NaNs(df):
            tqdm.write(OK)
        else:
            tqdm.write("Se han generado NANs!!" + FAIL)
            exit()

        tqdm.write(f"\tGrabando posicion de la estacion ", end='')
        df['longitud'] = estacion["longitud"]
        df['latitud'] = estacion["latitud"]
        df['altitud'] =estacion["altitud"]
        if no_NaNs(df):
            tqdm.write(OK)
        else:
            tqdm.write("Se han generado NANs!!" + FAIL)
            exit() 
        
        dfs_nombres.append(estacion["nombre"])
        dfs.append(df)
    
    tqdm.write(f"Organizando índices:", end='')
    fecha_min = None
    fecha_max = None
    for df in dfs:
        df['fecha'] = df.index
        if (fecha_max is None) or (df['fecha'].max() > fecha_max ):
            fecha_max = df['fecha'].max()
        if (fecha_min is None) or (df['fecha'].min() < fecha_min ):
            fecha_min = df['fecha'].min()
            
    tqdm.write(f"\tRango de fechas ({fecha_min}, {fecha_max}")
    for df in dfs:
        minimo = (df['fecha'].min() - fecha_min).days * 24 + (df['fecha'].min() - fecha_min).seconds / 3600
        maximo = (df['fecha'].max() - fecha_min).days * 24 + (df['fecha'].max() - fecha_min).seconds / 3600
        df.index = pd.RangeIndex(int(minimo), int(maximo) + 1)
        df.drop(columns=['fecha'], inplace=True)

    tqdm.write(f"\tActualizando distancias y altitudes de cada estacion", end='')
    cdg = distanciasCdG([x["latitud"] for x in estaciones if x["nombre"] in dfs_nombres],
                        [x["longitud"] for x in estaciones if x["nombre"] in dfs_nombres],
                        [x["altitud"] for x in estaciones if x["nombre"] in dfs_nombres])
    for df in dfs:
        df['distancia'] = haversine(lat1=df.iloc[0].loc['latitud'], long1=df.iloc[0].loc['longitud'], lat2=cdg[0], long2=cdg[1])
        df['altitud'] = df.iloc[0].loc['altitud'] - cdg[2]
        df.drop(columns=['longitud', 'latitud'], inplace=True)
        
    tqdm.write(OK)
    
    tqdm.write(f"\tAplicando escalado:", end='')
    
        # # escalar datos
        # pt = PowerTransformer(method='yeo-johnson', standardize=True)
        # df = pd.DataFrame(pt.fit_transform(df), columns=df.columns, index=df.index)
        # if no_NaNs(df):
        #     tqdm.write(OK)
        # else:
        #     tqdm.write("Se han generado NANs!!" + FAIL)
        #     exit()     
    parametros = dict()
    for metrica in tqdm((['temperatura', 'hr', 'precipitacion'] + var_bio + var_macd + var_nwp)):
        parametros[metrica] = dict()
        parametros[metrica]['max'] = float(max([_[metrica].max() for _ in dfs]))
        parametros[metrica]['min'] = float(min([_[metrica].min() for _ in dfs]))
        
        if metrica == 'nwp_temperatura':
            parametros[metrica]['max'] = parametros['temperatura']['max']
            parametros[metrica]['min'] = parametros['temperatura']['min']
        if metrica == 'nwp_hr':
            parametros[metrica]['max'] = parametros['hr']['max']
            parametros[metrica]['min'] = parametros['hr']['min']
        if metrica == 'nwp_precipitacion':
            parametros[metrica]['max'] = parametros['precipitacion']['max']
            parametros[metrica]['min'] = parametros['precipitacion']['min']
            
        scaler = Escalador(Xmax=parametros[metrica]['max'], Xmin=parametros[metrica]['min'], min=0, max=1, auto_scale=False)
        for df in dfs:
            df[metrica] = scaler.transform(np.array(df[metrica].values))
            if not no_NaNs(df):
                tqdm.write("Se han generado NANs!!" + FAIL)
                exit()
  
    tqdm.write(f"Generando metadatos ", end='')
    metadata = {}
    metadata["longitud"] = len(dfs)
    metadata["datasets"] = {}
    metadata["datasets"]["nombres"] = dfs_nombres
    metadata["datasets"]["longitud"] = [len(_) for _ in dfs]
    metadata["CdG"] = [float(_) for _ in list(cdg)]
    metadata['fecha_min'] = datetime.strftime(fecha_min, format="%Y-%m-%d %H:%M:%S")
    metadata['fecha_max'] = datetime.strftime(fecha_max, format="%Y-%m-%d %H:%M:%S")
    metadata['indice_min'] = min([_.index.min() for _ in dfs])
    metadata['indice_max'] = max([_.index.max() for _ in dfs])
    metadata['escaladores'] = {}
    metadata['escaladores'] = parametros
    tqdm.write(OK)
    
    return (dfs, metadata)

def generarvariablesPmodel(estacion:list=None, estaciones: list=None, escaladores: list=None, outliers: list=None, cfg: AttrDict=None) -> tuple:
    # ruta = Path(estacion["ruta"])
    # metricas = estacion["metricas"]
    # tqdm.write(Fore.YELLOW + f"{estacion['nombre']}" + Style.RESET_ALL)
    
    # tqdm.write(f"\tLeyendo estacion desde {ruta}", end='')
    # if not ruta.is_file():
    #     tqdm.write(f"No existe el archivo para {estacion['nombre']}")
    #     tqdm.write(FAIL)
    #     exit()
    
    # df = pd.read_csv(ruta, sep=',', decimal='.', header='infer')
    # df['fecha'] = pd.to_datetime(df['fecha'], format=estacion["format"])
    # df.set_index('fecha', drop=True, inplace=True)
    # tqdm.write(OK)
    
    # tqdm.write(f"\tComprobando variables de entrada", end='')
    # if not check_variables_entrada(df):
    #     exit()
        
    # tqdm.write(f"\tComprobando outliers", end='')
    # if not check_outliers(df, metricas, outliers):
    #     exit()
        
    # tqdm.write(f"\tResample a 1 hora", end='')
    # df = df.resample('1H').interpolate()
    # tqdm.write(OK)
  
    # tqdm.write(f"\tComprobando longitud del dataset (mínimo 30 días)", end='')
    # if not check_longitud(df, 30 * 24):
    #     exit()
    
    # tqdm.write(f"\tCalculando variables cíclicas de fecha", end='')
    # df['fecha'] = df.index
    # df['dia'], df['mes'], df['hour'] = temporales(df['fecha'])
    # df.drop(columns=['fecha'], inplace=True)
    # tqdm.write(OK)

    # tqdm.write(f"\tEliminando NaNs...", end='')
    # df = quitar_nans(df)
    # if df is None:
    #     exit()
    
    # tqdm.write(f"\tCalculando variables climáticas", end='')
    # var_bio, df = variables_bioclimaticas(df)
    # if var_bio is None:
    #     exit()

    # tqdm.write(f"\tCalculando MACD", end='')
    # var_macd, df = variables_macd(df)
    # if var_macd is None:
    #     exit()
    
    # tqdm.write(f"\tCalculando distancias y altitudes", end='')
    # with open(Path(cfg.paths.zmodel.dataset_metadata), 'r') as handler:
    #     metadata = yaml.safe_load(handler)
    # cdg = metadata["CdG"]
    # df['distancia'] = haversine(lat1=estacion['latitud'], long1=estacion['longitud'], lat2=cdg[0], long2=cdg[1])
    # df['altitud'] = estacion['altitud'] - cdg[2]
    # tqdm.write(OK)
 
    # tqdm.write(f"\tCalculando prediccion a nivel de zona", end='')
    # try:
    #     with open(Path(cfg.paths.zmodel.dataset), 'rb') as handler:
    #         datasets = pickle.load(handler)
    #     print(f"\t\tUsando {cfg.paths.zmodel.dataset} como archivo de datos procesados de estaciones")
    # except:
    #     print(Fore.RED + "Por favor defina un archivo de datos procesados" + Style.RESET_ALL)
    #     exit()
    # kwargs = {'datasets': datasets, 'fecha_inicio_test': df.index.min(), 'fecha_fin_test': df.index.max(),
    #           'fecha_inicio': datetime.strptime(metadata['fecha_min'], "%Y-%m-%d %H:%M:%S"), 
    #           'pasado': cfg.pasado, 'futuro': cfg.futuro, 'etiquetaX': cfg.prediccion, 'etiquetaF': list(cfg.zmodel.model.encoder.features),
    #           'etiquetaT': list(cfg.zmodel.model.decoder.features), 
    #           'name': cfg.experiment, 'model_name': cfg.zmodel.model.name,
    #           'rnn_num_layers': cfg.zmodel.model.encoder.rnn_num_layers, 'encoder_hidden_size': cfg.zmodel.model.encoder.hidden_size,
    #           'encoder_bidirectional' : cfg.zmodel.model.encoder.bidirectional, 'device': 'cuda' if cfg.zmodel.model.use_cuda else 'cpu',
    #           'encoder_rnn_dropout': cfg.zmodel.model.encoder.rnn_dropout, 'decoder_hidden_size': cfg.zmodel.model.decoder.hidden_size,
    #           'decoder_dropout': cfg.zmodel.model.decoder.dropout, 'model_scheduler': cfg.zmodel.model.scheduler, 'path_checkpoints': cfg.paths.zmodel.checkpoints,
    #           'use_checkpoint': cfg.zmodel.dataloaders.test.use_checkpoint, 'epochs': cfg.zmodel.model.epochs, 'path_model' : cfg.paths.zmodel.model,
    #           'indice_min' : metadata['indice_min'], 'indice_max' : metadata['indice_max']
    #           }
    # y_pred, _ = predictor.predict(**kwargs)
    # #y_pred = pickle.load(open('/home/diego/weather-control/data/processed/precalculadas_nivel_zona.pickle', 'rb'))
    # y_pred = np.array([_.mean(axis=0) for _ in y_pred])
    # y_pred = y_pred.squeeze()
    # y_pred = pd.DataFrame(y_pred)
    # y_pred.columns = ["Z"+str(_) for _ in y_pred.columns]
    # # concat necesita que los indices sean iguales
    # df.reset_index(inplace=True)
    # y_pred.reset_index(inplace=True)
    # df = pd.concat([df, y_pred], axis=1)
    # df.set_index('fecha', drop=True, inplace=True)
    # del y_pred
    # if not no_NaNs(df):
    #     print("Se han generado Nans" + FAIL)
    #     exit()
    # tqdm.write(OK)

    # tqdm.write(f"\tCalculando prediccion a nivel de la estacion", end='')
    # dfs_estacion, metadata_estacion = generarvariablesZmodel(estaciones=list(cfg.pmodel.estaciones), 
    #                        escaladores = cfg.preprocesado.escaladores, 
    #                        outliers = cfg.preprocesado.outliers)
    
    # tqdm.write(f"\t\tFecha de inicio de los datos de la estacion: {datetime.strftime(df.index.min(), '%Y-%m-%d %H:%M:%S')}")
    # fecha_inicio_prediccion = datetime.strftime(df.index.min() + timedelta(hours=cfg.pasado), '%Y-%m-%d %H:%M:%S')
    # tqdm.write(f"\t\tFecha de inicio de la prediccion : {fecha_inicio_prediccion}")
    # tqdm.write(f"\t\tFecha de fin de los datos de la estacion: {datetime.strftime(df.index.max(), '%Y-%m-%d %H:%M:%S')}")
    # fecha_fin_prediccion = datetime.strftime(df.index.max() - timedelta(hours=cfg.futuro), '%Y-%m-%d %H:%M:%S')
    # tqdm.write(f"\t\tFecha de fin de la prediccion : {fecha_fin_prediccion}")
    
    # kwargs = {'datasets': dfs_estacion, 'fecha_inicio_test': df.index.min() + timedelta(hours=cfg.pasado), 'fecha_fin_test': df.index.max() - timedelta(hours=cfg.futuro),
    #           'fecha_inicio': datetime.strptime(metadata_estacion['fecha_min'], "%Y-%m-%d %H:%M:%S"),
    #           'pasado': cfg.pasado, 'futuro': cfg.futuro, 'etiquetaX': cfg.prediccion, 'etiquetaF': list(cfg.zmodel.model.encoder.features), 'etiquetaT': list(cfg.zmodel.model.decoder.features),
    #           'name': cfg.experiment, 'model_name': cfg.zmodel.model.name, 
    #           'rnn_num_layers': cfg.zmodel.model.encoder.rnn_num_layers, 'encoder_hidden_size': cfg.zmodel.model.encoder.hidden_size,
    #           'encoder_bidirectional' : cfg.zmodel.model.encoder.bidirectional, 'device': 'cuda' if cfg.zmodel.model.use_cuda else 'cpu',
    #           'encoder_rnn_dropout': cfg.zmodel.model.encoder.rnn_dropout, 'decoder_hidden_size': cfg.zmodel.model.decoder.hidden_size,
    #           'decoder_dropout': cfg.zmodel.model.decoder.dropout, 'model_scheduler': cfg.zmodel.model.scheduler,
    #           'path_checkpoints': cfg.paths.zmodel.checkpoints, 'use_checkpoint': cfg.zmodel.dataloaders.test.use_checkpoint,
    #           'epochs': cfg.zmodel.model.epochs, 'path_model' : cfg.paths.zmodel.model,
    #           'indice_min' : metadata_estacion['indice_min'], 'indice_max' : metadata_estacion['indice_max']
    #           }
    # y_pred, _ = predictor.predict(**kwargs)
    # #y_pred = pickle.load(open('/home/diego/weather-control/data/processed/precalculadas_nivel_parcela.pickle', 'rb'))
    # y_pred = np.array([_.squeeze() for _ in y_pred])
    # y_pred = pd.DataFrame(y_pred)
    # y_pred.columns = ["P" + str(_) for _ in y_pred.columns]
    # # Es necesario cortar el dataset, ya que no tengo info sobre 
    # df = df.loc[(df.index >= df.index.min() + timedelta(hours=cfg.pasado)) & (df.index <= df.index.max() - timedelta(hours=cfg.futuro)), :]
    # # concat necesita que los indices sean iguales
    # df.reset_index(inplace=True)
    # y_pred.reset_index(inplace=True)
    # df = pd.concat([df, y_pred], axis=1)
    # df.set_index('fecha', drop=True, inplace=True)
    # del y_pred
    # if not no_NaNs(df):
    #     print("Se han generado Nans" + FAIL)
    #     exit()
    # tqdm.write(OK)

    # tqdm.write(f"\tObteniendo datos de AEMET", end='')
    # # son datos desde Najera y solo 30 horas
    # dfs_aemet, metadata_aemet = generarvariablesZmodel(estaciones=[{"nombre": "Nájera", "id": 1289, "format": "%Y-%m-%d %H:%M:%S",
    #                                                              "metricas": ["temperatura", "hr", "precipitacion"],
    #                                                              "ruta": "/home/diego/weather-control/data/processed/1289.csv",
    #                                                              "longitud": -2.7266667, "latitud": 42.4141667, "altitud": 500}], 
    #                        escaladores = cfg.preprocesado.escaladores, 
    #                        outliers = cfg.preprocesado.outliers)
    # df_aemet = dfs_aemet[0].loc[:, cfg.prediccion]
    # df_aemet = pd.DataFrame(df_aemet)
    # df_aemet.set_index(pd.date_range(start = metadata_aemet['fecha_min'],
    #                                  end=metadata_aemet['fecha_max'],
    #                                  freq='1H'), inplace=True)
    # df_aemet = df_aemet.loc[(df_aemet.index >= df.index.min()) & (df_aemet.index <= df.index.max()), : ]
    # data = extraccionFuturo(df_aemet, k=30)
    # data = pd.DataFrame(data)
    # data.columns = ["AE" + str(_) for _ in data.columns]
    # df.reset_index(inplace=True)
    # data.reset_index(inplace=True)
    # df = pd.concat([df, data], axis=1)
    # df.set_index('fecha', drop=True, inplace=True)
    # #print(data.info())
    # del data, df_aemet
    # # Recortar el dataset los 30 valores ultimos
    # df = df.iloc[0:-30, :]
    # if not no_NaNs(df):
    #     tqdm.write("Se han generado NANs" + FAIL)
    #     exit()
    # else:
    #     tqdm.write(OK)
    
    # tqdm.write(f"\tCalculando distancias y altitudes", end='')
    # df['distancia'] = haversine(lat1=estacion['latitud'], long1=estacion['longitud'], lat2=cdg[0], long2=cdg[1])
    # df['altitud'] = estacion['altitud'] - cdg[2]
    # tqdm.write(OK)

    
    # tqdm.write(f"\tAplicando escalado:", end='')
    # # escalar datos
    # for metrica in ['temperatura', 'hr', 'precipitacion']:
    #     escalador = escaladores[metrica]
    #     scaler = Escalador(Xmax=escalador['max'], Xmin=escalador['min'], min=-1, max=1, auto_scale=False)
    #     df[metrica] = scaler.transform(np.array(df[metrica].values))
    # for metrica in var_bio:
    #     scaler = Escalador(min=-1, max=1, auto_scale=True)
    #     df[metrica] = scaler.transform(np.array(df[metrica].values))
    # for metrica in var_macd:
    #     scaler = Escalador(min=-1, max=1, auto_scale=True)
    #     df[metrica] = scaler.transform(np.array(df[metrica].values))
    
    # if no_NaNs(df):
    #     tqdm.write(OK)
    # else:
    #     tqdm.write("Se han generado NANs!!" + FAIL)
    #     exit()
        
    # # tqdm.write(f"Generando metadatos ", end='')
    # # metadata = {}
    # # metadata['fecha_min'] = datetime.strftime(df.index.min(), format="%Y-%m-%d %H:%M:%S")
    # # metadata['fecha_max'] = datetime.strftime(df.index.max(), format="%Y-%m-%d %H:%M:%S")
    # # tqdm.write(OK)

    # return df
    pass
