"""
Diversas funciones para su uso dentro de la generacion de datasets tanto para zmodel y pmodel

Comun:
- distanciasCdG(latitudes: list, longitudes: list, altitudes: list) -> tuple
- haversine(lat1: float, long1: float, lat2: float, long2: float) -> float
- parser_experiment(cfg: dict, name: str) -> dict
- cyclic(column_date: pd.Series) -> tuple

Zmodel:
- generarvariablesZmodel(estaciones: list=None, escaladores: list=None, outliers: list=None) -> tuple

Pmodel:
- extraccionFuturo(df: pd.DataFrame, k: int=30 ) -> np.array
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from colorama import Fore, Back, Style
from math import sqrt, sin, cos, atan2, pi
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import common.utils.indicesbioclimaticos as bio
from common.utils.otrosindicadores import macd
from common.utils.scalers import Escalador

OK = "\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]"
FAIL = "\t[ " + Fore.RED + "FAIL" + Style.RESET_ALL + " ]"

def check_nan(df: pd.DataFrame=None) -> str:
    """ Chequea si un dataframe tiene nans"""
    
    if df.isna().sum().sum() > 0:
        return False
    else:
        return True

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

def parser_experiment(cfg: dict, name: str) -> dict:
    """Parsea la cadena {{experiment}} de los archivos de configuracion yml

    Args:
        cfg (dict): diccionario
        name (str): reemplazo

    Returns:
        dict: diccionario parseado
    """
    for key in cfg.keys():
        if isinstance(cfg[key], dict):
            cfg[key] = parser_experiment(cfg[key], name)
        if isinstance(cfg[key], str):
            cfg[key] = cfg[key].replace("{{experiment}}", name)
    return cfg

def cyclic(column_date: pd.DataFrame) -> tuple:
    """Generador de variables de fecha ciclicas

    Args:
        df (pd.Series): Columna en formato datetime para extraer las variables físicas

    Returns:
        tuple: (día_sin, dia_cos, diasemana_sin, diasemana_cos, mes_sin, mes_cos)
    """
    df = pd.DataFrame()

    df['dia'] = column_date.apply(lambda x: x.day * 2 * pi / 31)
    df['dia_sin'] = np.sin(df['dia'])
    df['dia_cos'] = np.cos(df['dia'])

    df['diasemana'] = column_date.apply(lambda x: x.isoweekday() * 2 * pi  / 7)
    df['diasemana_sin'] = np.sin(df['diasemana'])
    df['diasemana_cos'] = np.cos(df['diasemana'])

    df['mes'] = column_date.apply(lambda x: x.month * 2 * pi  / 12)
    df['mes_sin'] = np.sin(df['mes'])
    df['mes_cos'] = np.cos(df['mes'])

    return (df['dia_sin'], df['dia_cos'], df['diasemana_sin'], df['diasemana_cos'], df['mes_sin'], df['mes_cos'])

def generarvariablesZmodel(estaciones: list=None, escaladores: list=None, outliers: list=None) -> tuple:
    
    dfs = []
    dfs_nombres = []
    for estacion in tqdm(estaciones, desc='Estaciones'):
        ruta = Path(estacion["ruta"])
        metricas = estacion["metricas"]
        
        tqdm.write(Fore.YELLOW + f"{estacion['nombre']}" + Style.RESET_ALL)
        tqdm.write(f"\tLeyendo estacion desde {ruta}", end='')
    
        if not ruta.is_file():
            tqdm.write(Fore.RED + f"No existe el archivo para {estacion['nombre']}" + Style.RESET_ALL)
            tqdm.write(FAIL)
            continue
        else:
            df = pd.read_csv(ruta, sep=',', decimal='.', header='infer')
            df['fecha'] = pd.to_datetime(df['fecha'], format=estacion["format"] if "format" in estacion.keys() else "%Y-%m-%d %H:%M:%S")
            df.set_index(df['fecha'], drop=True, inplace=True)
            tqdm.write(OK)
        
        tqdm.write(f"\tComprobando variables de entrada", end='')
        if 'temperatura' not in df.columns or 'hr' not in df.columns or 'precipitacion' not in df.columns:
            tqdm.write(FAIL)
            continue
        else:
            tqdm.write(OK)

        tqdm.write(f"\tComprobando outliers", end='')
        for i, metrica in enumerate(metricas):
            outlier = outliers[metrica]
            if metrica in df.columns:
                assert not any(df[metrica] > outlier['max'] * (1 + outlier['tol'] / 100)), Fore.RED + f"Valores de {metrica} superan el valor de {outlier['max']}" + Style.RESET_ALL
                assert not any(df[metrica] < outlier['min'] * (1 + outlier['tol'] / 100)), Fore.RED + f"Valores de {metrica} superan el valor de {outlier['min']}" + Style.RESET_ALL
        tqdm.write(OK)

        tqdm.write(f"\tResample a 1 hora", end='')
        df = df.resample('1H').mean()
        tqdm.write(OK)
        
        tqdm.write(f"\tComprobando longitud del dataset mayor de un año", end='')
        if len(df) > 365 * 24:
            tqdm.write(OK)
        else:
             tqdm.write(FAIL)
             continue
        
        tqdm.write(f"\tCalculando variables cíclicas de fecha", end='')
        df['fecha'] = df.index
        df['dia_sin'], df['dia_cos'], df['diasemana_sin'], df['diasemana_cos'], df['mes_sin'], df['mes_cos'] = cyclic(df['fecha'])
        df.drop(columns=['fecha'], inplace=True)
        tqdm.write(OK)
        
        tqdm.write(f"\tEliminando NaNs...", end='')
        if df.isna().sum().sum() > 0:
            tqdm.write(f"{df.isna().sum().sum()}", end='')
            imp_mean = IterativeImputer(random_state=0)
            idf = pd.DataFrame(imp_mean.fit_transform(df), columns=df.columns)
            df['temperatura'] = idf['temperatura'].values
            df['hr'] = idf['hr'].values
            df['precipitacion'] = idf['precipitacion'].values
            #df.interpolate(inplace=True)  # eliminar NaNs
            del idf
        tqdm.write(" -> " + str(df.isna().sum().sum()), end='')
        tqdm.write("\t" + OK)
        if not check_nan(df):
            tqdm.write(Fore.RED + "\tQuedan NaNs en el dataset. Por favor compruebe" + Style.RESET_ALL)
            #tqdm.write(Fore.YELLOW + "\tForzando la eliminacion" + Style.RESET_ALL)
            #df.dropna(inplace=True)
            exit()
        ###TODO CAMBIAR ESTO PORQUE PERDEMOS TIEMPO
        df.drop(columns=['dia_sin', 'dia_cos', 'diasemana_sin', 'diasemana_cos', 'mes_sin', 'mes_cos'], inplace=True)

        tqdm.write(f"\tCalculando variables climáticas", end='')
        if set(['temperatura']).issubset(set(df.columns)): 
            df['biotemperatura'] = bio.biotemperatura(temperatura=np.array(df['temperatura'].values), muestras_dia=24)
            assert df['biotemperatura'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['biotemperatura']" + Style.RESET_ALL
        if set(['temperatura']).issubset(set(df.columns)): 
            df['integraltermica'] = bio.integral_termica(temperatura=np.array(df['temperatura'].values))
            assert df['integraltermica'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['integraltermica']" + Style.RESET_ALL
        if set(['precipitacion']).issubset(set(df.columns)): 
            df['integralrain'] = bio.integral_precipitacion(arr=np.array(df['precipitacion'].values,), periodo=3 * 30)
            assert df['integralrain'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['integralrain']" + Style.RESET_ALL
        if set(['temperatura', 'precipitacion']).issubset(set(df.columns)): 
            df['indicelasser'] = bio.indice_lasser(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
            assert df['indicelasser'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicelasser']" + Style.RESET_ALL
        if set(['temperatura', 'precipitacion']).issubset(set(df.columns)): 
            df['indicelang'] = bio.indice_lang(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
            assert df['indicelang'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicelang']" + Style.RESET_ALL
        if set(['temperatura', 'precipitacion']).issubset(set(df.columns)): 
            df['indiceangtrom'] = bio.indice_angtrom(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
            assert df['indiceangtrom'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['angtrom']" + Style.RESET_ALL
        if set(['temperatura', 'precipitacion']).issubset(set(df.columns)): 
            df['indicegasparin'] = bio.indice_gasparin(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
            assert df['indicegasparin'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicegasparin']" + Style.RESET_ALL
        if set(['temperatura', 'precipitacion']).issubset(set(df.columns)): 
            df['indicemartonne'] = bio.indice_martonne(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
            assert df['indicemartonne'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicemartonne']" + Style.RESET_ALL
        if set(['temperatura', 'precipitacion']).issubset(set(df.columns)):     
            df['indicebirot'] = bio.indice_birot(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
            assert df['indicebirot'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicebirot']" + Style.RESET_ALL
        if set(['temperatura']).issubset(set(df.columns)): 
            df['indicehugling'] = bio.indice_hugling(temperatura=np.array(df['temperatura'].values),  muestras_dia=24)
            assert df['indicehugling'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicehugling']" + Style.RESET_ALL
        if check_nan(df):
            tqdm.write(OK)
        else:
            tqdm.write("Se han generado NANs!!" + FAIL)
            exit()
        
        tqdm.write(f"\tCalculando MACD", end='')
        if set(['temperatura']).issubset(set(df.columns)):
            df['tempmacd'], df['tempmacdsenal'] = macd(np.array(df['temperatura'].values))
            assert df['tempmacd'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['tempmacd']" + Style.RESET_ALL
            assert df['tempmacdsenal'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['tempmacdsenal']" + Style.RESET_ALL
        if set(['hr']).issubset(set(df.columns)):
            df['hrmacd'], df['hrmacdsenal'] = macd(np.array(df['hr'].values))
            assert df['hrmacd'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['hrmacd']" + Style.RESET_ALL
            assert df['hrmacdsenal'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['hrmacdsenal']" + Style.RESET_ALL
        if set(['precipitacion']).issubset(set(df.columns)):
            df['precipitacionmacd'], df['precipitacionmacdsenal'] = macd(np.array(df['precipitacion'].values))
            assert df['precipitacionmacd'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['precipitacionmacd']" + Style.RESET_ALL
            assert df['precipitacionmacdsenal'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['precipitacionmacdsenal']" + Style.RESET_ALL
        if check_nan(df):
            tqdm.write(OK)
        else:
            tqdm.write("Se han generado NANs!!" + FAIL)
            exit()
        
        #ESTA PARTE DEBE REALIZARSE MEJOR PORQUE ESCALA TODO Y HAY QUE DECIRLE QUE ESCALAR        
        tqdm.write(f"\tAplicando escalado:", end='')
        for metrica in df.columns:
            if metrica in metricas:
                escalador = escaladores[metrica]
                scaler = Escalador(Xmax=escalador['max'], 
                                   Xmin=escalador['min'],
                                   min=-1,
                                   max=1,
                                   auto_scale=False)
            else: # es metrica derivada
                scaler = Escalador(min=-1,
                                   max=1,
                                   auto_scale=True)
            df[metrica] = scaler.transform(np.array(df[metrica].values))                
        if check_nan(df):
            tqdm.write(OK)
        else:
            tqdm.write("Se han generado NANs!!" + FAIL)
            exit()     

        tqdm.write(f"\tCalculando variables cíclicas de fecha", end='')
        df['fecha'] = df.index
        df['dia_sin'], df['dia_cos'], df['diasemana_sin'], df['diasemana_cos'], df['mes_sin'], df['mes_cos'] = cyclic(df['fecha'])
        if check_nan(df):
            tqdm.write(OK)
        else:
            tqdm.write("Se han generado NANs!!" + FAIL)
            exit()

        tqdm.write(f"\tGrabando posicion de la estacion ", end='')
        df['longitud'] = estacion["longitud"]
        df['latitud'] = estacion["latitud"]
        df['altitud'] =estacion["altitud"]
        if check_nan(df):
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
        if (fecha_max is None) or (df['fecha'].max() > fecha_max ):
            fecha_max = df['fecha'].max()
        if (fecha_min is None) or (df['fecha'].min() < fecha_min ):
            fecha_min = df['fecha'].min()
            
    tqdm.write(f"\tRango de fechas ({fecha_min}, {fecha_max}")
    for i, df in enumerate(dfs):
        minimo = (df['fecha'].min() - fecha_min).days * 24 + (df['fecha'].min() - fecha_min).seconds / 3600
        maximo = (df['fecha'].max() - fecha_min).days * 24 + (df['fecha'].max() - fecha_min).seconds / 3600
        # tqdm.write(f"len: {len(df)}")
        # tqdm.write(f"\t{df['fecha'].min()} - {df['fecha'].max()}")
        # tqdm.write(f"\t{i}: {min} - {max}")
        df.index = pd.RangeIndex(int(minimo), int(maximo) + 1)
        # tqdm.write(f"\t{df.index}")
        df.drop(columns=['fecha'], inplace=True)

    tqdm.write(f"\tActualizando distancias y altitudes de cada estacion", end='')
    cdg = distanciasCdG([x["latitud"] for x in estaciones if x["nombre"] in dfs_nombres],
                        [x["longitud"] for x in estaciones if x["nombre"] in dfs_nombres],
                        [x["altitud"] for x in estaciones if x["nombre"] in dfs_nombres])
    
    for df in dfs:
        df['distancia'] = haversine(lat1=df.iloc[0].loc['latitud'],
                                    long1=df.iloc[0].loc['longitud'],
                                    lat2=cdg[0],
                                    long2=cdg[1])
        df['altitud'] = df.iloc[0].loc['altitud'] - cdg[2]
        df.drop(columns=['longitud', 'latitud'], inplace=True)
    tqdm.write(OK)
    
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
    tqdm.write(OK)
    
    return (dfs, metadata)

def extraccionFuturo(df: pd.DataFrame, k: int=30 ) -> np.array:
    data = np.empty(shape=(len(df) , k))  #ojo la longitud real es len(df) - k pero 
    # lo saco con la misma longitud que el df para que los pueda juntar los dataset
    for i in range(len(df) - k):
        for j in range(k):
            data[i, j] = df.iloc[i + j + 1, 0]
    return data