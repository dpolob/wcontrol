import click
import yaml
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from attrdict import AttrDict
from tqdm import tqdm
from colorama import Fore, Back, Style
from pathlib import Path

import torch
from common.utils.hparams import load_hyperparameter

import common.predict.modules as predictor
from common.utils.parser import parser


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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
    name = cfg["experiment"]
    epoch = cfg["zmodel"]["dataloaders"]["test"]["use_checkpoint"]
    cfg = AttrDict(parser(name, epoch)(cfg))

    try:
        with open(Path(cfg.paths.zmodel.dataset), 'rb') as handler:
            datasets = pickle.load(handler)
        print(f"Usando {cfg.paths.zmodel.dataset} como archivo de datos procesados de estaciones")
    except:
        print(Fore.RED + "Por favor defina un archivo de datos procesados")
        exit()
    
    if not cfg.zmodel.dataloaders.test.enable:
        print("El archivo no tiene definido dataset para test")
        exit()
    
    with open(Path(cfg.paths.zmodel.dataset_metadata), 'r') as handler:
        metadata = yaml.safe_load(handler)
    print("Leidos metadatos del dataset")
    print(f"Inicio del dataset en {metadata['fecha_min']}")
    
    kwargs_dataloader = {
        'datasets': datasets,
        'fecha_inicio_test': datetime.strptime(cfg.zmodel.dataloaders.test.fecha_inicio, "%Y-%m-%d %H:%M:%S"),
        'fecha_fin_test': datetime.strptime(cfg.zmodel.dataloaders.test.fecha_fin, "%Y-%m-%d %H:%M:%S"),
        'fecha_inicio': datetime.strptime(metadata['fecha_min'], "%Y-%m-%d %H:%M:%S"),
        'pasado': cfg.pasado,
        'futuro': cfg.futuro,
        'etiquetaX': list(cfg.prediccion),
        'etiquetaF': list(cfg.zmodel.model.encoder.features),
        'etiquetaT': list(cfg.zmodel.model.decoder.features),
        'etiquetaP': list(cfg.zmodel.model.decoder.nwp),
        'indice_max': metadata['indice_max'],
        'indice_min': metadata['indice_min']
        }
    
    kwargs_prediccion= {
        'name': name,
        'device': 'cuda' if cfg.zmodel.model.use_cuda else 'cpu',
        'path_checkpoints': cfg.paths.zmodel.checkpoints,
        'use_checkpoint': cfg.zmodel.dataloaders.test.use_checkpoint,
        'path_model' : cfg.paths.zmodel.model,
        } 
 
    test_dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
    y_pred = predictor.predict(test_dataloader, **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
    
    assert y_pred.shape[0]==len(test_dataloader), "Revisar y_pred y_pred.shape[0]!!!"
    assert y_pred.shape[3]==len(list(cfg.prediccion)), "Revisar y_pred.shape[3]!!!"
    assert y_pred.shape[2]==cfg.futuro, "Revisar y_pred.shape[2]!!!"
   
    # Creamos la matriz y de salida real, con el mismo shape que las predicciones
    y_real = np.empty_like(y_pred)
    y_nwp = np.empty_like(y_pred)
    for i, (_, _, _, Y, P) in enumerate(tqdm(test_dataloader)):
        # hay que quitarles la componente 0 y pasarlos a numpy
        y_real[i, ...] = Y[:, 1:, :].numpy()
        y_nwp[i, ...] = P[:, 1:, :].numpy()
        
    predicciones = {'y_pred': y_pred, 'y_real': y_real, 'y_nwp': y_nwp}
   
    output = Path(cfg.paths.zmodel.predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Usando {output} como ruta para guardar predicciones")
    with open(output, 'wb') as handler:
        pickle.dump(predicciones, handler)
    print(f"Salvando archivo de predicciones en {output}")

@cli.command()                                           
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
def pmodel(file):
    pass

if __name__ == "__main__":
    cli()