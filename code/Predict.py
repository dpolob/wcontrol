import click
import yaml
import pickle
import torch
import copy
import numpy as np
from datetime import datetime
from attrdict import AttrDict
from tqdm import tqdm
from colorama import Fore
from pathlib import Path
from torch.utils.data import DataLoader

import common.predict.modules as predictor
from common.utils.parser import parser
from common.utils.kwargs_gen import generar_kwargs
from common.utils.datasets import dataset_pmodel as ds_pmodel
from common.utils.datasets import sampler_pmodel as sa_pmodel

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(420)
np.random.seed(420)
OK = "\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]"
FAIL = "\t[ " + Fore.RED + "FAIL" + Style.RESET_ALL + " ]"

@click.group()
def main():
    pass

@main.command()                                           
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
def zmodel(file):

    try:
        with open(file, 'r') as handler:
            cfg = yaml.safe_load(handler)
            name = cfg["experiment"]
            epoch = cfg["zmodel"]["dataloaders"]["test"]["use_checkpoint"]
            cfg = AttrDict(parser(name, epoch)(cfg))
        print(f"Usando {file} como archivo de configuracion")
        with open(Path(cfg.paths.zmodel.dataset), 'rb') as handler:
            datasets = pickle.load(handler)
        print(f"Usando {cfg.paths.zmodel.dataset} como archivo de datos procesados de estaciones")
        with open(Path(cfg.paths.zmodel.dataset_metadata), 'r') as handler:
            metadata = yaml.safe_load(handler)
        print("Leidos metadatos del dataset")
    except Exception as e:
        print(f"El archivo de configuracion del experimento no existe o no existe el archivo {cfg.paths.zmodel.dataset} \
            con los datasets para el modelo zonal o {cfg.paths.zmodel.dataset_metadata} de metadatos del dataset del \
            modelo zonal. Mas info: {e}")
        exit()
    
    if not cfg.zmodel.dataloaders.test.enable:
        print("El archivo no tiene definido dataset para test")
        exit()
    
    print(f"Inicio del dataset en {metadata['fecha_min']}")
    kwargs_dataloader = generar_kwargs()._dataloader(modelo='zmodel', fase='test', cfg=cfg, datasets=datasets, metadata=metadata)
    kwargs_prediccion = generar_kwargs()._predict(modelo='zmodel', cfg=cfg)
 
    test_dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
    y_pred = predictor.predict(test_dataloader, **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
    y_real = np.empty_like(y_pred)
    assert y_pred.shape[0]==len(test_dataloader), "Revisar y_pred y_pred.shape[0]!!!"
    assert y_pred.shape[3]==len(list(cfg.prediccion)), "Revisar y_pred.shape[3]!!!"
    assert y_pred.shape[2]==cfg.futuro, "Revisar y_pred.shape[2]!!!"
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
        print(f"El archivo de configuracion del experimento no existe o no existe el archivo {cfg.paths.pmodel.dataset} \
            con el dataset para el modelo de parcela o {cfg.paths.pmodel.dataset_metadata} de metadatos del dataset del \
            modelo de parcela. Mas info en: {e}")
        exit()
    
    
    device = 'cuda' if cfg.pmodel.model.use_cuda else 'cpu'
    Fout = len(cfg.prediccion)
    
    ## Generar Dataset de test para acelerar el proceso de entrenamiento
    print("Generando datos temporales para acelerar el proceso de predicion")
    try:
        test_dataset = pickle.load(open(Path(cfg.paths.pmodel.pmodel_test), "rb"))
        print(f"\tYa existen los datos temporales... Cargango datos de test desde el modelo pmodel desde {cfg.paths.pmodel.pmodel_test}")
    except (OSError, IOError) as e:
        print(f"\tGenerando datos de test...")
        kwargs_dataloader = generar_kwargs()._dataloader(modelo='pmodel', fase='test', cfg=cfg, datasets=datasets, metadata=metadata)
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
        print(f"No se ha definido que predecir. Ver pmodel --help {FAIL}")
        exit()
    if temp:
        ## Parte especifica temperatura
        print("Prediccion modelo de temperatura...")
        cfg = AttrDict(parser(None, None, 'temperatura')(copy.deepcopy(cfg_previo)))
        test_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=test_dataset, componentes=slice(0, 1)),
                                        sampler=sa_pmodel.PModelSampler(datasets=test_dataset, batch_size=1, shuffle=False),    
                                        batch_size=None,
                                        num_workers=2)
       
        print(f"\tDataset de test: {len(test_dataloader)}", end='')
        kwargs_prediccion = generar_kwargs._predict(modelo='pmodel', cfg=cfg)
        y_pred = predictor.predict(test_dataloader, tipo='pmodel', **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
        y_pred = y_pred.squeeze(axis=1)  # (len(test), N, Ly, Fout).squeeze(axis=1) -> (len(test), Ly, Fout)
        y_real = np.empty_like(y_pred)
        y_nwp = np.empty_like(y_pred)
        for i, (X, Y) in enumerate(tqdm(test_dataloader)):
            y_real[i, ...] = Y[0, :, :].numpy()
            y_nwp[i,...] = X[0, :, :].numpy()
        predicciones = {'y_pred': y_pred, 'y_real': y_real, 'y_nwp': y_nwp}
        print(OK)    
        output = Path(cfg.paths.pmodel.predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"\tUsando {output} como ruta para guardar predicciones", end='')
        with open(output, 'wb') as handler:
            pickle.dump(predicciones, handler)
        print(OK)

    if hr:
        ## Parte especifica temperatura
        print("Prediccion modelo de humedad...")
        cfg = AttrDict(parser(None, None, 'hr')(copy.deepcopy(cfg_previo)))
        test_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=test_dataset, componentes=slice(1, 2)),
                                        sampler=sa_pmodel.PModelSampler(datasets=test_dataset, batch_size=1, shuffle=False),    
                                        batch_size=None,
                                        num_workers=2)
       
        print(f"\tDataset de test: {len(test_dataloader)}", end='')
        kwargs_prediccion = generar_kwargs._predict(modelo='pmodel', cfg=cfg) 
        y_pred = predictor.predict(test_dataloader, tipo='pmodel', **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
        y_pred = y_pred.squeeze(axis=1)  # (len(test), N, Ly, Fout).squeeze(axis=1) -> (len(test), Ly, Fout)
        y_real = np.empty_like(y_pred)
        y_nwp = np.empty_like(y_pred)
        for i, (X, Y) in enumerate(tqdm(test_dataloader)):
            y_real[i, ...] = Y[0, :, :].numpy()
            y_nwp[i,...] = X[0, :, :].numpy()
        predicciones = {'y_pred': y_pred, 'y_real': y_real, 'y_nwp': y_nwp}
        print(OK)
        output = Path(cfg.paths.pmodel.predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"\tUsando {output} como ruta para guardar predicciones", end='')
        with open(output, 'wb') as handler:
            pickle.dump(predicciones, handler)
        print(OK)
        
    if rain:
        ## Parte especifica temperatura
        print("Prediccion modelo de precipitacion...")
        cfg = AttrDict(parser(None, None, 'precipitacion')(copy.deepcopy(cfg_previo)))
        test_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=test_dataset, componentes=slice(2, 2 + len(metadata["bins"]))),
                                        sampler=sa_pmodel.PModelSampler(datasets=test_dataset, batch_size=1, shuffle=False),    
                                        batch_size=None,
                                        num_workers=2)
       
        print(f"\tDataset de test: {len(test_dataloader)}", end='')    
        kwargs_prediccion = generar_kwargs._predict(modelo='pmodel', cfg=cfg) 
        y_pred = predictor.predict(test_dataloader, tipo='pmodel', **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
        y_pred = y_pred.squeeze(axis=1)  # (len(test), N, Ly, Fout).squeeze(axis=1) -> (len(test), Ly, Fout)
        y_real = np.empty_like(y_pred)
        y_nwp = np.empty_like(y_pred)
        for i, (X, Y) in enumerate(tqdm(test_dataloader)):
            y_real[i, ...] = Y[0, :, :].numpy()
            y_nwp[i,...] = X[0, :, :].numpy()
        predicciones = {'y_pred': y_pred, 'y_real': y_real, 'y_nwp': y_nwp}
        print(OK)
        output = Path(cfg.paths.pmodel.predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"\tUsando {output} como ruta para guardar predicciones", end='')
        with open(output, 'wb') as handler:
            pickle.dump(predicciones, handler)
        print(OK)

if __name__ == "__main__":
    main()