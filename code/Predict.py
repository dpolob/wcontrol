import click
import yaml
import pickle
import numpy as np
from datetime import datetime
from attrdict import AttrDict
from tqdm import tqdm
from colorama import Fore, Back, Style
from pathlib import Path


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import common.predict.modules as predictor
import copy

from common.utils.parser import parser
from common.utils.kwargs_gen import generar_kwargs
from common.utils.datasets import dataset_seq2seq as ds
from common.utils.datasets import dataset_pmodel as ds_pmodel
from common.utils.datasets import sampler_seq2seq as sa
from common.utils.datasets import sampler_pmodel as sa_pmodel
from common.utils.trainers import trainerSeq2Seq as tr
from common.utils.trainers import trainerpmodel as tr_pmodel
from common.utils.loss_functions import lossfunction as lf
import models.pmodel.p_model as md_pmodel
import importlib


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

@click.group()
def main():
    pass

@main.command()                                           
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
    y_real = np.empty_like(y_pred)
    assert y_pred.shape[0]==len(test_dataloader), "Revisar y_pred y_pred.shape[0]!!!"
    assert y_pred.shape[3]==len(list(cfg.prediccion)), "Revisar y_pred.shape[3]!!!"
    assert y_pred.shape[2]==cfg.futuro, "Revisar y_pred.shape[2]!!!"
   
    # Creamos la matriz y de salida real, con el mismo shape que las predicciones
    
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

@main.command()                                           
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
@click.option('--temp', is_flag=True, default= False, help='Predecir modelo de temperatura')
@click.option('--hr', is_flag=True, default= False, help='Predecir modelo de HR')
@click.option('--rain', is_flag=True, default= False, help='Predecir modelo de precipitacion')
def pmodel(file, temp, hr, rain):
    
    try:
        with open(file, 'r') as handler:
            cfg = yaml.safe_load(handler)
            name = cfg["experiment"]
            epoch = cfg["pmodel"]["dataloaders"]["test"]["use_checkpoint"]
            cfg = AttrDict(parser(name, epoch)(cfg))    
        with open(Path(cfg.paths.pmodel.dataset), 'rb') as handler:
            datasets = pickle.load(handler)
        with open(Path(cfg.paths.pmodel.dataset_metadata), 'r') as handler:
            metadata = yaml.safe_load(handler)
    except Exception as e:
        print(f"Algun archivo no puede ser leido. {e}")
    device = 'cuda' if cfg.pmodel.model.use_cuda else 'cpu'
    Fout = len(cfg.prediccion)
    
    ## Generar Dataset de test
    try:
        test_dataset = pickle.load(open(Path(cfg.paths.pmodel.pmodel_test), "rb"))
        print(f"Cargango datos de test desde el modelo pmodel desde {cfg.paths.pmodel.pmodel_test}")
    except (OSError, IOError) as e:
        print(f"Generando datos de test...")
        kwargs_dataloader = generar_kwargs()._dataloader(model='pmodel', fase='test', cfg=cfg, datasets=datasets, metadata=metadata)
        dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
        y_real = np.empty((len(dataloader), cfg.futuro, Fout))
        x_nwp = np.empty_like(y_real)
        for i, (_, _, _, Y, P) in enumerate(tqdm(dataloader)):
            # hay que quitarles la componente 0 y pasarlos a numpy porque son tensores
            y_real[i, ...] = Y[:, 1:, :].numpy()  # y_real = (len(dataset), Ly, Fout)
            x_nwp[i, ...] = P[:, 1:, :].numpy()  # pred_nwp = (len(dataset), Ly, Fout)
        test_dataset = [x_nwp, y_real]
        with open(Path(cfg.paths.pmodel.pmodel_test), 'wb') as handler:
            pickle.dump(test_dataset, handler)
    
    cfg_previo = copy.deepcopy(dict(cfg))
    if not temp and not hr and not rain:
        print("No se ha definido que predecir. Ver pmodel --help")
        exit()
    if temp:
        ## Parte especifica temperatura
        print("Prediccion modelo de temperatura...")
        cfg = AttrDict(parser(None, None, 'temperatura')(copy.deepcopy(cfg_previo)))
        test_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=test_dataset, componentes=slice(0, 1)),
                                        sampler=sa_pmodel.PModelSampler(datasets=test_dataset, batch_size=1, shuffle=False),    
                                        batch_size=None,
                                        num_workers=2)
       
        print(f"\tDataset de test: {len(test_dataloader)}")

        kwargs_prediccion= {
            'name': name,
            'device': device,
            'path_checkpoints': cfg.paths.pmodel.checkpoints,
            'use_checkpoint': cfg.pmodel.dataloaders.test.use_checkpoint,
            'path_model' : cfg.paths.pmodel.model,
            } 
        y_pred = predictor.predict(test_dataloader, tipo='pmodel', **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
        y_pred = y_pred.squeeze(axis=1)  # (len(test), N, Ly, Fout).squeeze(axis=1) -> (len(test), Ly, Fout)
        y_real = np.empty_like(y_pred)
        y_nwp = np.empty_like(y_pred)
        for i, (X, Y) in enumerate(tqdm(test_dataloader)):
            y_real[i, ...] = Y[0, :, :].numpy()
            y_nwp[i,...] = X[0, :, :].numpy()
        predicciones = {'y_pred': y_pred, 'y_real': y_real, 'y_nwp': y_nwp}
    
        output = Path(cfg.paths.pmodel.predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Usando {output} como ruta para guardar predicciones")
        with open(output, 'wb') as handler:
            pickle.dump(predicciones, handler)
        print(f"Salvando archivo de predicciones en {output}")

    if hr:
        ## Parte especifica temperatura
        print("Prediccion modelo de humedad...")
        cfg = AttrDict(parser(None, None, 'hr')(copy.deepcopy(cfg_previo)))
        test_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=test_dataset, componentes=slice(1, 2)),
                                        sampler=sa_pmodel.PModelSampler(datasets=test_dataset, batch_size=1, shuffle=False),    
                                        batch_size=None,
                                        num_workers=2)
       
        print(f"\tDataset de test: {len(test_dataloader)}")
        kwargs_prediccion= {
            'name': name,
            'device': device,
            'path_checkpoints': cfg.paths.pmodel.checkpoints,
            'use_checkpoint': cfg.pmodel.dataloaders.test.use_checkpoint,
            'path_model' : cfg.paths.pmodel.model,
            } 
        y_pred = predictor.predict(test_dataloader, tipo='pmodel', **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
        y_pred = y_pred.squeeze(axis=1)  # (len(test), N, Ly, Fout).squeeze(axis=1) -> (len(test), Ly, Fout)
        y_real = np.empty_like(y_pred)
        y_nwp = np.empty_like(y_pred)
        for i, (X, Y) in enumerate(tqdm(test_dataloader)):
            y_real[i, ...] = Y[0, :, :].numpy()
            y_nwp[i,...] = X[0, :, :].numpy()
        predicciones = {'y_pred': y_pred, 'y_real': y_real, 'y_nwp': y_nwp}
    
        output = Path(cfg.paths.pmodel.predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Usando {output} como ruta para guardar predicciones")
        with open(output, 'wb') as handler:
            pickle.dump(predicciones, handler)
        print(f"Salvando archivo de predicciones en {output}")
    
        output = Path(cfg.paths.pmodel.predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Usando {output} como ruta para guardar predicciones")
        with open(output, 'wb') as handler:
            pickle.dump(predicciones, handler)
        print(f"Salvando archivo de predicciones en {output}")
    if rain:
        ## Parte especifica temperatura
        print("Prediccion modelo de precipitacion...")
        cfg = AttrDict(parser(None, None, 'precipitacion')(copy.deepcopy(cfg_previo)))
        test_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=test_dataset, componentes=slice(2, 2 + len(metadata["bins"]))),
                                        sampler=sa_pmodel.PModelSampler(datasets=test_dataset, batch_size=1, shuffle=False),    
                                        batch_size=None,
                                        num_workers=2)
       
        print(f"\tDataset de test: {len(test_dataloader)}")    
    
        kwargs_prediccion= {
            'name': name,
            'device': device,
            'path_checkpoints': cfg.paths.pmodel.checkpoints,
            'use_checkpoint': cfg.pmodel.dataloaders.test.use_checkpoint,
            'path_model' : cfg.paths.pmodel.model,
            }  
        y_pred = predictor.predict(test_dataloader, tipo='pmodel', **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
        y_pred = y_pred.squeeze(axis=1)  # (len(test), N, Ly, Fout).squeeze(axis=1) -> (len(test), Ly, Fout)
        y_real = np.empty_like(y_pred)
        y_nwp = np.empty_like(y_pred)
        for i, (X, Y) in enumerate(tqdm(test_dataloader)):
            y_real[i, ...] = Y[0, :, :].numpy()
            y_nwp[i,...] = X[0, :, :].numpy()
        predicciones = {'y_pred': y_pred, 'y_real': y_real, 'y_nwp': y_nwp}
    
        output = Path(cfg.paths.pmodel.predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Usando {output} como ruta para guardar predicciones")
        with open(output, 'wb') as handler:
            pickle.dump(predicciones, handler)
        print(f"Salvando archivo de predicciones en {output}")

if __name__ == "__main__":
    main()