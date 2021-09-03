#%%

# DONE
# Revision de outliers en temperatura, humedad y precipitacion
# máxima registrada en La Rioja de 42.8 grados el 7 julio de 1982 y el de agosto que de momento es de 40.6. 
# minima -9.8 en Logroño => -20ºC aviso solo

# Hi.
# I have noticed the operation here ( features = (features - 0.5)*2) in Generative Adversarial Networks (GAN). I don't understand why we need to do this here. The mean and variance of MINIST dataset are 0.1307 and 0.3081. Can you please explain the meaning of doing so? Looking forward to your reply.

# —
# You are receiving this because you are subscribed to this thread.
# Reply to this email directly, view it on GitHub, or unsubscribe.
# Triage notifications on the go with GitHub Mobile for iOS or Android.


# Sebastian Raschka
# jue, 5 ago 15:01
# Good question. Which notebook is that? My spontaneous thought is that I probably did that because PyTorch's data transformation normalizes pixels to [0, 1] rang

# annyWangAn <notifications@github.com>
# mié, 11 ago 13:06 (hace 8 días)
# para rasbt/deeplearning-models, Subscribed

# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan/gan.ipynb
# Why we need the images in [-1,1] range? What is the difference between the ranges of [0,1] and [-1,1]? Does the image range have a big influence on the net performance? Didn’t see a similar operation in the previous network. So why in this net we need to do so? Thank you again for your reply.



# Sebastian Raschka <notifications@github.com>
# mié, 11 ago 14:47 (hace 8 días)
# para rasbt/deeplearning-models, Subscribed

# Usually gradient descent behaves a bit better if the values are centered at 0. (Ideally, the mean should be zero). In practice, I don't notice big differences though to be honest.



import click
import yaml
import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from tqdm import tqdm
from colorama import Fore, Back, Style
from math import pi,sqrt,sin,cos,atan2
from attrdict import AttrDict

import common.utils.indicesbioclimaticos as bio
from common.utils.scalers import Escalador
from common.utils.otrosindicadores import macd

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

@click.command()
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
def preprocesado(file):
    """
    Script para el preprocesado de datos de las estaciones.
    Las estaciones se especifican en el archivo de configuracion .yml
    """

    try:
        with open(file, 'r') as handler:
            cfg = AttrDict(yaml.safe_load(handler))
        print(f"Usando {file} como archivo de configuracion")
    except:
        print(f"{file} no existe. Por favor defina un archivo con --file")
        exit()
        
    if 'datasets' not in cfg.paths.keys() or 'preprocesado' not in cfg.keys() or \
        'estaciones' not in cfg.keys():
        print(Fore.RED + f"{file} no contine informacion básica")
    else:
        output = Path(cfg.paths.datasets)
        output.parent.mkdir(parents=True, exist_ok=True)
        
    estaciones = list(cfg.estaciones)
    escaladores = cfg.preprocesado.escaladores
    outliers = cfg.preprocesado.outliers
    
    cdg = distanciasCdG([x['latitud'] for x in estaciones], 
                        [x['longitud'] for x in estaciones],
                        [x['altitud'] for x in estaciones])


    dfs = []
    for estacion in tqdm(estaciones, desc='Estaciones'):
        ruta = Path(estacion["ruta"])
        metricas = estacion["metricas"]
        
        print(Fore.YELLOW + f"{estacion['nombre']}" + Style.RESET_ALL)
        print(f"\tLeyendo estacion desde {ruta}", end='')
    
        if not ruta.is_file():
            print(Fore.RED + f"No existe el archivo para {estacion['nombre']}" + Style.RESET_ALL)
            print("\t[ " + Fore.RED + "FAIL" + Style.RESET_ALL + " ]")
            continue
        else:
            df = pd.read_csv(ruta, sep=',', decimal='.', header='infer')
            df['fecha'] = pd.to_datetime(df['fecha'], format=estacion["format"])
            df.set_index(df['fecha'], drop=True, inplace=True)
            print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
        
        print(f"\tComprobando variables de entrada", end='')
        if 'temperatura' not in df.columns or 'hr' not in df.columns or 'precipitacion' not in df.columns:
            print("\t[ " + Fore.RED + "FAIL" + Style.RESET_ALL + " ]")
            continue
        else:
            print("\t\t\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")

        print(f"\tComprobando outliers", end='')
        for i, metrica in enumerate(metricas):
            outlier = outliers[metrica]
            if metrica in df.columns:
                assert not any(df[metrica] > outlier['max'] * (1 + outlier['tol'] / 100)), Fore.RED + f"Valores de {metrica} superan el valor de {outlier['max']}" + Style.RESET_ALL
                assert not any(df[metrica] < outlier['min'] * (1 + outlier['tol'] / 100)), Fore.RED + f"Valores de {metrica} superan el valor de {outlier['min']}" + Style.RESET_ALL
        print("\t\t\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")

        print(f"\tResample a 1 hora", end='')
        df = df.resample('1H').mean()
        print("\t\t\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
        
        print(f"\tComprobando longitud del dataset", end='')
        if len(df) > 365 * 24:
            print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
        else:
             print("\t[ " + Fore.RED +"FAIL" + Style.RESET_ALL + " ]")
             continue

        print(f"\tEliminando NaNs...", end='')
        print(df.isna().sum().sum(), end='')
        df.interpolate(inplace=True)  # eliminar NaNs
        print(" -> " + str(df.isna().sum().sum()), end='')
        print("\t\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
        if df.isna().sum().sum() != 0:
            print(Fore.RED + "\tQuedan NaNs en el dataset. Por favor compruebe" + Style.RESET_ALL)
            print(Fore.YELLOW + "\tForzando la eliminacion" + Style.RESET_ALL)
            df.dropna(inplace=True)
          

        print(f"\tCalculando variables climáticas", end='')
        if set(['temperatura']).issubset(set(df.columns)): 
            df['biotemperatura'] = bio.biotemperatura(temperatura=np.array(df['temperatura'].values),
                                                    muestras_dia=24)
            assert df['biotemperatura'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['biotemperatura']" + Style.RESET_ALL 
        
        if set(['temperatura']).issubset(set(df.columns)): 
            df['integraltermica'] = bio.integral_termica(temperatura=np.array(df['temperatura'].values))
            assert df['integraltermica'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['integraltermica']" + Style.RESET_ALL
        
        if set(['temperatura', 'precipitacion']).issubset(set(df.columns)): 
            df['indicelasser'] = bio.indice_lasser(precipitacion=np.array(df['precipitacion'].values),
                                               temperatura=np.array(df['temperatura'].values),
                                               muestras_dia=24)
            assert df['indicelasser'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicelasser']" + Style.RESET_ALL
            
        if set(['temperatura', 'precipitacion']).issubset(set(df.columns)): 
            df['indicelang'] = bio.indice_lang(precipitacion=np.array(df['precipitacion'].values),
                                            temperatura=np.array(df['temperatura'].values),
                                            muestras_dia=24)
            assert df['indicelang'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicelang']" + Style.RESET_ALL
            
        if set(['temperatura', 'precipitacion']).issubset(set(df.columns)): 
            df['indiceangtrom'] = bio.indice_angtrom(precipitacion=np.array(df['precipitacion'].values),
                                                    temperatura=np.array(df['temperatura'].values),
                                                    muestras_dia=24)
            assert df['indiceangtrom'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['angtrom']" + Style.RESET_ALL
            
        if set(['temperatura', 'precipitacion']).issubset(set(df.columns)): 
            df['indicegasparin'] = bio.indice_gasparin(precipitacion=np.array(df['precipitacion'].values),
                                                    temperatura=np.array(df['temperatura'].values),
                                                    muestras_dia=24)
            assert df['indicegasparin'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicegasparin']" + Style.RESET_ALL
        
        if set(['temperatura', 'precipitacion']).issubset(set(df.columns)): 
            df['indicemartonne'] = bio.indice_martonne(precipitacion=np.array(df['precipitacion'].values),
                                                       temperatura=np.array(df['temperatura'].values),
                                                       muestras_dia=24)
            assert df['indicemartonne'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicemartonne']" + Style.RESET_ALL
        
        if set(['temperatura', 'precipitacion']).issubset(set(df.columns)):     
            df['indicebirot'] = bio.indice_birot(precipitacion=np.array(df['precipitacion'].values),
                                                temperatura=np.array(df['temperatura'].values),
                                                muestras_dia=24)
            assert df['indicebirot'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicebirot']" + Style.RESET_ALL
            
        if set(['temperatura']).issubset(set(df.columns)): 
            df['indicehugling'] = bio.indice_hugling(temperatura=np.array(df['temperatura'].values),
                                                    muestras_dia=24)
            assert df['indicehugling'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicehugling']" + Style.RESET_ALL
            
        print("\t\t[ " + Fore.GREEN + "OK" + Style.RESET_ALL + " ]")
        
        print(f"\tCalculando MACD", end='')
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
        print("\t\t\t\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
        
        # print("bio: ", df.biotemperatura.max(), df.biotemperatura.min())
        # print("df['integraltermica']", df['integraltermica'].max(), df['integraltermica'].min())
        # print("df['indicelasser']", df['indicelasser'].max(), df['indicelasser'].min())
        # print("df['indicelang']", df['indicelang'].max(), df['indicelang'].min())
        # print("df['indiceangtrom']", df['indiceangtrom'].max(), df['indiceangtrom'].min())
        # print("df['indicegasparin']", df['indicegasparin'].max(), df['indicegasparin'].min())
        # print("df['indicemartonne']", df['indicemartonne'].max(), df['indicemartonne'].min())
        # print("df['indicebirot']", df['indicebirot'].max(), df['indicebirot'].min())        
        # print("df['indicehugling']", df['indicehugling'].max(), df['indicehugling'].min())
        # print("df['tempmacd']", df['tempmacd'].max(), df['tempmacd'].min())
        # print("df['tempmacdsenal']", df['tempmacdsenal'].max(), df['tempmacdsenal'].min())
        # print("df['hrmacd']", df['hrmacd'].max(), df['hrmacd'].min())
        # print("df['hrmacdsenal']", df['hrmacdsenal'].max(), df['hrmacdsenal'].min())
        # print("df['precipitacionmacd']", df['precipitacionmacd'].max(), df['precipitacionmacd'].min())
        # print("df['precipitacionmacdsenal']", df['precipitacionmacdsenal'].max(), df['precipitacionmacdsenal'].min())
                
        print(f"\tAplicando escalado:", end='')
        # escalar variables
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
        
        print("\t\t\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
        assert df.isna().sum().sum() == 0, Fore.RED + "Existen NANs!!!" + Style.RESET_ALL        

        print(f"\tCalculando variables cíclicas de fecha", end='')
        df['fecha'] = df.index
        df['dia'] = df['fecha'].apply(lambda x: x.day * 2 * pi / 31)
        df['dia_sin'] = np.sin(df['dia'])
        df['dia_cos'] = np.cos(df['dia'])

        df['diasemana'] = df['fecha'].apply(lambda x: x.isoweekday() * 2 * pi  / 7)
        df['diasemana_sin'] = np.sin(df['diasemana'])
        df['diasemana_cos'] = np.cos(df['diasemana'])

        df['mes'] = df['fecha'].apply(lambda x: x.month * 2 * pi  / 12)
        df['mes_sin'] = np.sin(df['mes'])
        df['mes_cos'] = np.cos(df['mes'])

        df.drop(columns=['dia', 'diasemana', 'mes'], inplace=True)
        print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")

        print(f"\tCalculando distancias y altitudes", end='')
        df['distancia'] = haversine(lat1=estacion['latitud'],
                                    long1=estacion['longitud'],
                                    lat2=cdg[0],
                                    long2=cdg[1])
        
        df['altitud'] = estacion['altitud'] - cdg[2]
        print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")

        dfs.append(df)
        
    print(f"Organizando índices:", end='')
    fecha_min = None
    fecha_max = None
    for df in dfs:
        if (fecha_max is None) or (df['fecha'].max() > fecha_max ):
            fecha_max = df['fecha'].max()
        if (fecha_min is None) or (df['fecha'].min() < fecha_min ):
            fecha_min = df['fecha'].min()
             
    print(f"\tRango de fechas ({fecha_min}, {fecha_max}")
    for i, df in enumerate(dfs):
        min = (df['fecha'].min() - fecha_min).days * 24 + (df['fecha'].min() - fecha_min).seconds / 3600
        max = (df['fecha'].max() - fecha_min).days * 24 + (df['fecha'].max() - fecha_min).seconds / 3600
        # print(f"len: {len(df)}")
        # print(f"\t{df['fecha'].min()} - {df['fecha'].max()}")
        # print(f"\t{i}: {min} - {max}")
        
        df.index = pd.RangeIndex(int(min), int(max) + 1)
        # print(f"\t{df.index}")
        df.drop(columns=['fecha'], inplace=True)
                       
    print(f"Guardando salida en {output}", end='')
    with open(output, 'wb') as handler:
        pickle.dump(dfs, handler)

    print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")

if __name__ == "__main__":
    preprocesado()

# %%
