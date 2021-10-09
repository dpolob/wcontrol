#%%

import click
import yaml
import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from attrdict import AttrDict
from datetime import datetime, timedelta

from colorama import Fore, Back, Style

from common.data_preprocessing.modules import parser_experiment
from common.data_preprocessing.modules import haversine, distanciasCdG
from Zmodel_predict import predictor
from common.data_preprocessing.modules import generarvariablesZmodel, extraccionFuturo

import common.utils.indicesbioclimaticos as bio
from common.utils.otrosindicadores import macd
from common.utils.scalers import Escalador
from common.data_preprocessing.modules import cyclic

@click.group()
def cli():
    pass

@cli.command()
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
def zmodel(file):
    try:
        with open(file, 'r') as handler:
            cfg = yaml.safe_load(handler)
        print(f"Usando {file} como archivo de configuracion")
    except:
        print(f"{file} no existe. Por favor defina un archivo con --file")
        exit()
    
    name = cfg["experiment"]
    cfg = AttrDict(parser_experiment(cfg, name)) # parser de {{experiment}}
    dfs, metadata = generarvariablesZmodel(estaciones=list(cfg.zmodel.estaciones), 
                           escaladores = cfg.preprocesado.escaladores, 
                           outliers = cfg.preprocesado.outliers)
    
    
    print(f"Guardando salida en {cfg.paths.zmodel.dataset}", end='')
    with open(cfg.paths.zmodel.dataset, 'wb') as handler:
        pickle.dump(dfs, handler)
    print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")                             
    with open(Path(cfg.paths.zmodel.dataset_metadata), 'w') as handler:
        yaml.safe_dump(metadata, handler, allow_unicode=True)
    print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
    
@cli.command()
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
def pmodel(file):
    """

    """
    try:
        with open(file, 'r') as handler:
            cfg = yaml.safe_load(handler)
        print(f"Usando {file} como archivo de configuracion")
    except:
        print(f"{file} no existe. Por favor defina un archivo con --file")
        exit()
    name = cfg["experiment"]
    #parser de {{experiment}}
    cfg = AttrDict(parser_experiment(cfg, name))
    
    # output = Path(cfg.paths.pmodel.dataset)
    # output.parent.mkdir(parents=True, exist_ok=True)
    estacion = list(cfg.pmodel.estaciones)[0]
    #estaciones = list(cfg.zmodel.estaciones)
    escaladores = cfg.preprocesado.escaladores
    outliers = cfg.preprocesado.outliers
    
    ruta = Path(estacion["ruta"])
    metricas = estacion["metricas"]
        
    print(Fore.YELLOW + f"{estacion['nombre']}" + Style.RESET_ALL)
    print(f"\tLeyendo estacion desde {ruta}", end='')
    
    if not ruta.is_file():
        print(Fore.RED + f"No existe el archivo para {estacion['nombre']}" + Style.RESET_ALL)
        print("\t[ " + Fore.RED + "FAIL" + Style.RESET_ALL + " ]")
        exit()
    
    df = pd.read_csv(ruta, sep=',', decimal='.', header='infer')

    df['fecha'] = pd.to_datetime(df['fecha'], format=estacion["format"])
    df.set_index('fecha', drop=True, inplace=True)
    print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
    
    
    print(f"\tComprobando variables de entrada", end='')
    if 'temperatura' not in df.columns or 'hr' not in df.columns or 'precipitacion' not in df.columns:
        print("\t[ " + Fore.RED + "FAIL" + Style.RESET_ALL + " ]")
        exit()
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
  
    print(f"\tComprobando longitud del dataset (mínimo 30 días)", end='')
    if len(df) > 30 * 24:
        print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
    else:
        print("\t[ " + Fore.RED +"FAIL" + Style.RESET_ALL + " ]")
        exit()

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
    variables_bioclimaticas =[]
    if set(['temperatura']).issubset(set(df.columns)): 
        df['integraltermica'] = bio.integral_termica(temperatura=np.array(df['temperatura'].values))
        variables_bioclimaticas.append('integraltermica')
        assert df['integraltermica'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['integraltermica']" + Style.RESET_ALL
        
        df['indicehugling'] = bio.indice_hugling(temperatura=np.array(df['temperatura'].values), muestras_dia=24)
        variables_bioclimaticas.append('indicehugling')
        assert df['indicehugling'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicehugling']" + Style.RESET_ALL
    
    if set(['temperatura', 'precipitacion']).issubset(set(df.columns)): 
        df['indicelasser'] = bio.indice_lasser(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
        variables_bioclimaticas.append('indicelasser')
        assert df['indicelasser'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicelasser']" + Style.RESET_ALL
        
        df['indiceangtrom'] = bio.indice_angtrom(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
        variables_bioclimaticas.append('indiceangtrom')
        assert df['indiceangtrom'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['angtrom']" + Style.RESET_ALL
        
        df['indicemartonne'] = bio.indice_martonne(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values),muestras_dia=24)
        variables_bioclimaticas.append('indicemartonne')
        assert df['indicemartonne'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicemartonne']" + Style.RESET_ALL
    
        df['indicebirot'] = bio.indice_birot(precipitacion=np.array(df['precipitacion'].values), temperatura=np.array(df['temperatura'].values), muestras_dia=24)
        variables_bioclimaticas.append('indicebirot')
        assert df['indicebirot'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['indicebirot']" + Style.RESET_ALL
    print("\t\t[ " + Fore.GREEN + "OK" + Style.RESET_ALL + " ]")
    
    print(f"\tCalculando MACD", end='')
    variables_macd = []
    if set(['temperatura']).issubset(set(df.columns)):
        df['tempmacd'], df['tempmacdsenal'] = macd(np.array(df['temperatura'].values))
        variables_macd.append('tempmacd')
        variables_macd.append('tempmacdsenal')
        assert df['tempmacd'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['tempmacd']" + Style.RESET_ALL
        assert df['tempmacdsenal'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['tempmacdsenal']" + Style.RESET_ALL
        
    if set(['hr']).issubset(set(df.columns)):
        df['hrmacd'], df['hrmacdsenal'] = macd(np.array(df['hr'].values))
        variables_macd.append('hrmacd')
        variables_macd.append('hrmacdsenal')
        assert df['hrmacd'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['hrmacd']" + Style.RESET_ALL
        assert df['hrmacdsenal'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['hrmacdsenal']" + Style.RESET_ALL
    
    if set(['precipitacion']).issubset(set(df.columns)):
        df['precipitacionmacd'], df['precipitacionmacdsenal'] = macd(np.array(df['precipitacion'].values))
        variables_macd.append('precipitacionmacd')
        variables_macd.append('precipitacionmacdsenal')
        assert df['precipitacionmacd'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['precipitacionmacd']" + Style.RESET_ALL
        assert df['precipitacionmacdsenal'].isna().sum() == 0, Fore.RED + "Existen NaNs!! en df['precipitacionmacdsenal']" + Style.RESET_ALL
    print("\t\t\t\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
    
    print(f"\tCalculando distancias y altitudes", end='')
    with open(Path(cfg.paths.zmodel.dataset_metadata), 'r') as handler:
        metadata = yaml.safe_load(handler)
    cdg = metadata["CdG"]
    df['distancia'] = haversine(lat1=estacion['latitud'], long1=estacion['longitud'], lat2=cdg[0],long2=cdg[1])
    df['altitud'] = estacion['altitud'] - cdg[2]
    print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
    
    print(f"\tCalculando variables cíclicas de fecha", end='')
    df['fecha'] = df.index
    df['dia_sin'], df['dia_cos'], df['diasemana_sin'], df['diasemana_cos'], df['mes_sin'], df['mes_cos'] = cyclic(df['fecha'])
    df.drop(columns=['fecha'], inplace=True)
    print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")

    print(f"\tCalculando prediccion a nivel de zona", end='')

    try:
        with open(Path(cfg.paths.zmodel.dataset), 'rb') as handler:
            datasets = pickle.load(handler)
        print(f"\t\tUsando {cfg.paths.zmodel.dataset} como archivo de datos procesados de estaciones")
    except:
        print(Fore.RED + "Por favor defina un archivo de datos procesados" + Style.RESET_ALL)
        exit()
    

    kwargs = {'datasets': datasets,
              'fecha_inicio_test': df.index.min(),
        'fecha_fin_test': df.index.max(),
      'fecha_inicio': datetime.strptime(metadata['fecha_min'], "%Y-%m-%d %H:%M:%S"),
      'pasado': cfg.pasado,
      'futuro': cfg.futuro,
      'etiquetaX': cfg.prediccion,
      'etiquetaF': list(cfg.zmodel.model.encoder.features),
      'etiquetaT': list(cfg.zmodel.model.decoder.features),
      'name': name,
      'model_name': cfg.zmodel.model.name,
      'rnn_num_layers': cfg.zmodel.model.encoder.rnn_num_layers,
      'encoder_hidden_size': cfg.zmodel.model.encoder.hidden_size,
      'encoder_bidirectional' : cfg.zmodel.model.encoder.bidirectional,
      'device': 'cuda' if cfg.zmodel.model.use_cuda else 'cpu',
      'encoder_rnn_dropout': cfg.zmodel.model.encoder.rnn_dropout,
      'decoder_hidden_size': cfg.zmodel.model.decoder.hidden_size,
      'decoder_dropout': cfg.zmodel.model.decoder.dropout,
      'model_scheduler': cfg.zmodel.model.scheduler,
      'path_checkpoints': cfg.paths.zmodel.checkpoints,
      'use_checkpoint': cfg.zmodel.dataloaders.test.use_checkpoint,
      'epochs': cfg.zmodel.model.epochs,
      'path_model' : cfg.paths.zmodel.model,
    'indice_min' : metadata['indice_min'],
    'indice_max' : metadata['indice_max'],
        }
    # y_pred, _ = predictor(**kwargs)
    y_pred = pickle.load(open('/home/diego/weather-control/experiments/experimento7/predi__afff.pickle', 'rb'))
    y_pred = np.array([_.mean(axis=0) for _ in y_pred])
    y_pred = y_pred.squeeze()

    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = ["Z"+str(_) for _ in y_pred.columns]
    # concat necesita que los indices sean iguales
    df.reset_index(inplace=True)
    y_pred.reset_index(inplace=True)
    df = pd.concat([df, y_pred], axis=1)
    df.set_index('fecha', drop=True, inplace=True)
    del y_pred
    print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")

    print(f"\tCalculando prediccion a nivel de la estacion", end='')

    dfs_estacion, metadata_estacion = generarvariables(estaciones=list(cfg.pmodel.estaciones), 
                           escaladores = cfg.preprocesado.escaladores, 
                           outliers = cfg.preprocesado.outliers)
    
    print(f"\t\tFecha de inicio de los datos de la estacion: {datetime.strftime(df.index.min(), '%Y-%m-%d %H:%M:%S')}")
    fecha_inicio_prediccion = datetime.strftime(df.index.min() + timedelta(hours=cfg.pasado), '%Y-%m-%d %H:%M:%S')
    print(f"\t\tFecha de inicio de la prediccion : {fecha_inicio_prediccion}")
    print(f"\t\tFecha de fin de los datos de la estacion: {datetime.strftime(df.index.max(), '%Y-%m-%d %H:%M:%S')}")
    fecha_fin_prediccion = datetime.strftime(df.index.max() - timedelta(hours=cfg.futuro), '%Y-%m-%d %H:%M:%S')
    print(f"\t\tFecha de fin de la prediccion : {fecha_fin_prediccion}")
    
    kwargs = {'datasets': dfs_estacion,
              'fecha_inicio_test': df.index.min() + timedelta(hours=cfg.pasado),
        'fecha_fin_test': df.index.max() - timedelta(hours=cfg.futuro),
      'fecha_inicio': datetime.strptime(metadata_estacion['fecha_min'], "%Y-%m-%d %H:%M:%S"),
      'pasado': cfg.pasado,
      'futuro': cfg.futuro,
      'etiquetaX': cfg.prediccion,
      'etiquetaF': list(cfg.zmodel.model.encoder.features),
      'etiquetaT': list(cfg.zmodel.model.decoder.features),
      'name': name,
      'model_name': cfg.zmodel.model.name,
      'rnn_num_layers': cfg.zmodel.model.encoder.rnn_num_layers,
      'encoder_hidden_size': cfg.zmodel.model.encoder.hidden_size,
      'encoder_bidirectional' : cfg.zmodel.model.encoder.bidirectional,
      'device': 'cuda' if cfg.zmodel.model.use_cuda else 'cpu',
      'encoder_rnn_dropout': cfg.zmodel.model.encoder.rnn_dropout,
      'decoder_hidden_size': cfg.zmodel.model.decoder.hidden_size,
      'decoder_dropout': cfg.zmodel.model.decoder.dropout,
      'model_scheduler': cfg.zmodel.model.scheduler,
      'path_checkpoints': cfg.paths.zmodel.checkpoints,
      'use_checkpoint': cfg.zmodel.dataloaders.test.use_checkpoint,
      'epochs': cfg.zmodel.model.epochs,
      'path_model' : cfg.paths.zmodel.model,
    'indice_min' : metadata_estacion['indice_min'],
    'indice_max' : metadata_estacion['indice_max'],
        }
    #y_pred, _ = predictor(**kwargs)
    y_pred = pickle.load(open('/home/diego/weather-control/experiments/experimento7/predi__2709.pickle', 'rb'))
    y_pred = np.array([_.squeeze() for _ in y_pred])
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = ["P" + str(_) for _ in y_pred.columns]
    # Es necesario cortar el dataset, ya que no tengo info sobre 
    df = df.loc[(df.index >= df.index.min() + timedelta(hours=cfg.pasado)) & (df.index <= df.index.max() - timedelta(hours=cfg.futuro)), :]
    # concat necesita que los indices sean iguales
    df.reset_index(inplace=True)
    y_pred.reset_index(inplace=True)
    df = pd.concat([df, y_pred], axis=1)
    df.set_index('fecha', drop=True, inplace=True)
    del y_pred
    print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")

    print(f"\tObteniendo datos de AEMET", end='')
    # son datos desde Najera y solo 30 horas
    dfs_aemet, metadata_aemet = generarvariables(estaciones=[{"nombre": "Nájera", "id": 1289, "format": "%Y-%m-%d %H:%M:%S",
                                                                 "metricas": ["temperatura", "hr", "precipitacion"],
                                                                 "ruta": "/home/diego/weather-control/data/processed/1289.csv",
                                                                 "longitud": -2.7266667, "latitud": 42.4141667, "altitud": 500}], 
                           escaladores = cfg.preprocesado.escaladores, 
                           outliers = cfg.preprocesado.outliers)
    #df_aemet = pd.read_csv('/home/diego/weather-control/data/processed/1289.csv', sep=',', decimal='.', header='infer')
    #df_aemet['fecha'] = pd.to_datetime(df_aemet['fecha'], format="%Y-%m-%d %H:%M:%S")
    #df_aemet.set_index(df_aemet['fecha'], drop=True, inplace=True)
    df_aemet = dfs_aemet[0].loc[:, cfg.prediccion]
    df_aemet = pd.DataFrame(df_aemet)
    df_aemet.set_index(pd.date_range(start = metadata_aemet['fecha_min'],
                                     end=metadata_aemet['fecha_max'],
                                     freq='1H'), inplace=True)
    
    df_aemet = df_aemet.loc[(df_aemet.index >= df.index.min()) & (df_aemet.index <= df.index.max()), : ]
    data = extraccionFuturo(df_aemet, k=30)
    data = pd.DataFrame(data)
    data.columns = ["AE" + str(_) for _ in data.columns]
    
    df.reset_index(inplace=True)
    data.reset_index(inplace=True)
    df = pd.concat([df, data], axis=1)
    df.set_index('fecha', drop=True, inplace=True)
    print(data.info())
    del data, df_aemet
    # Recortar el dataset los K valores ultimos
    df = df.iloc[0:-30, :]
    print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
    
    
    print(df.info())
    
    exit()
    
    
    
    #### AL FINAL 
    print(f"\tAplicando escalado:", end='')
    # escalar datos
    for metrica in ['temperatura', 'hr', 'precipitacion']:
        escalador = escaladores[metrica]
        scaler = Escalador(Xmax=escalador['max'], 
                                Xmin=escalador['min'],
                                min=-1,
                                max=1,
                                auto_scale=False)
        df[metrica] = scaler.transform(np.array(df[metrica].values))
    for metrica in variables_bioclimaticas:
        scaler = Escalador(min=-1,
                                   max=1,
                                   auto_scale=True)
        df[metrica] = scaler.transform(np.array(df[metrica].values))
    for metrica in variables_macd:
        scaler = Escalador(min=-1,
                                   max=1,
                                   auto_scale=True)
        df[metrica] = scaler.transform(np.array(df[metrica].values))                  
        
    
    
    print("\t\t\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
    assert df.isna().sum().sum() == 0, Fore.RED + "Existen NANs!!!" + Style.RESET_ALL   
        
    
    print("\t\t\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")
    assert df.isna().sum().sum() == 0, Fore.RED + "Existen NANs!!!" + Style.RESET_ALL        



    print(f"\tCalculando distancias y altitudes", end='')
    df['distancia'] = haversine(lat1=estacion['latitud'], long1=estacion['longitud'], lat2=cdg[0], long2=cdg[1])
    df['altitud'] = estacion['altitud'] - cdg[2]
    print("\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]")

    
if __name__ == "__main__":
    cli()


# %%
