# %%
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import click
import yaml
import copy
from pathlib import Path
from attrdict import AttrDict
from tqdm import tqdm
from typing import Union

from common.utils.parser import parser


def generar_muestras(muestras: Union[str, list], inicio: int, fin: Union[str, int], paso: int , y_pred: np.ndarray ) -> list:
        
    f = len(y_pred) if fin == 'end' else fin
    if isinstance(muestras, list):
        return muestras
    elif muestras == 'range':  # range
        p = paso
        return range(inicio, f, p)
    elif muestras == 'random':
        return list(np.random.choice(range(inicio, fin), size= paso, replace=False))
    elif muestras == 'all':
        f = len(y_pred)
        return(inicio, f)
    else:
        raise NotImplementedError


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

    print(f"Usando {cfg.paths.zmodel.predictions} como archivo de predicciones")
    with open(Path(cfg.paths.zmodel.predictions), 'rb') as handler:
        predicciones = pickle.load(handler)
        print("Cargado archivo de predicciones")
    y_pred, y_real, y_nwp = predicciones['y_pred'], predicciones['y_real'], predicciones['y_nwp']


    ##############################################
    #### COSAS QUE HACER AQUI               ######
    ##############################################
    
    if isinstance(cfg.zmodel.resultados.visualizacion.muestras, str):  # all, random o range
        if cfg.zmodel.resultados.visualizacion.muestras == 'range':  # range
            inicio = cfg.zmodel.resultados.visualizacion.inicio
            fin = len(y_pred) if cfg.zmodel.resultados.visualizacion.fin == 'end' else cfg.zmodel.resultados.visualizacion.fin
            paso = cfg.zmodel.resultados.visualizacion.paso
            muestras = range(inicio, fin, paso)
    
    plots_path = Path(cfg.paths.zmodel.viz) 
    plots_path.mkdir(parents=True, exist_ok=True)   
    

    for idx in tqdm(muestras):
        for prediccion in cfg.zmodel.resultados.visualizacion.prediccion:
            if prediccion == 'temperatura':
                plt.plot(np.mean(y_pred[idx, ..., 0], axis=0).reshape(-1,1), 'red')
                plt.plot(np.mean(y_real[idx, ..., 0], axis=0).reshape(-1,1), 'green')
                plt.plot(np.mean(y_nwp[idx,...,0], axis=0).reshape(-1,1), 'magenta')
            elif prediccion == 'hr':
                plt.plot(np.mean(y_pred[idx, ..., 1], axis=0).reshape(-1,1), 'red')
                plt.plot(np.mean(y_real[idx, ..., 1], axis=0).reshape(-1,1), 'green')
                plt.plot(np.mean(y_nwp[idx,...,1], axis=0).reshape(-1,1), 'magenta')
            elif prediccion == 'precipitacion':
                plt.plot(np.mean(np.argmax(y_pred[idx,...,2:], axis=-1), axis=0).reshape(-1,1), 'red')
                plt.plot(np.mean(np.argmax(y_real[idx,...,2:], axis=-1), axis=0).reshape(-1,1), 'green')
                plt.plot(np.mean(np.argmax(y_nwp[idx,...,2:], axis=-1), axis=0).reshape(-1,1), 'magenta')
                  
            plt.savefig(plots_path / f"{idx}_{prediccion}.png", dpi=300)
            plt.close()

@main.command()                                           
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
@click.option('--temp', is_flag=True, default= False, help='Visualizar modelo de temperatura')
@click.option('--hr', is_flag=True, default= False, help='Visualizar modelo de HR')
@click.option('--rain', is_flag=True, default= False, help='Visualizar modelo de precipitacion')
def pmodel(file, temp, hr, rain):
    
    try:
        with open(file, 'r') as handler:
            cfg = yaml.safe_load(handler)
        print(f"Usando {file} como archivo de configuracion")
    except:
        print(f"{file} no existe. Por favor defina un archivo con --file")
    
    name = cfg["experiment"]
    epoch = cfg["pmodel"]["dataloaders"]["test"]["use_checkpoint"]
    cfg = AttrDict(parser(name, epoch)(cfg))


    ##############################################
    #### COSAS QUE HACER AQUI               ######
    ##############################################
    #GENERAR MUESTRAS
    
    cfg_previo = copy.deepcopy(dict(cfg))
    if not temp and not hr and not rain:
        print("No se ha definido que predecir. Ver pmodel --help")
        exit()
   
    if temp:
        print("Generando resultados de predicciones para temperatura")
        cfg = AttrDict(parser(None, None, 'temperatura')(copy.deepcopy(cfg_previo)))
        plots_path = Path(cfg.paths.pmodel.viz) 
        plots_path.mkdir(parents=True, exist_ok=True)
        print(f"\tUsando {cfg.paths.pmodel.predictions} como archivo de predicciones")
        with open(Path(cfg.paths.pmodel.predictions), 'rb') as handler:
            predicciones = pickle.load(handler)
        y_pred, y_real, y_nwp = predicciones['y_pred'], predicciones['y_real'], predicciones['y_nwp']
        muestras = generar_muestras(**cfg.pmodel.resultados, y_pred=y_pred)
        for idx in tqdm(muestras):
            plt.plot(y_pred[idx,...].reshape(-1,1), 'red')
            plt.plot(y_real[idx,...].reshape(-1,1), 'green')
            plt.plot(y_nwp[idx,...].reshape(-1,1), 'magenta')
            plt.savefig(plots_path / f"{idx}.png", dpi=300)
            plt.close()
    if hr:
        print("Generando resultados de predicciones para hr")
        cfg = AttrDict(parser(None, None, 'hr')(copy.deepcopy(cfg_previo)))
        plots_path = Path(cfg.paths.pmodel.viz) 
        plots_path.mkdir(parents=True, exist_ok=True)
        print(f"\tUsando {cfg.paths.pmodel.predictions} como archivo de predicciones")
        with open(Path(cfg.paths.pmodel.predictions), 'rb') as handler:
            predicciones = pickle.load(handler)
        y_pred, y_real, y_nwp = predicciones['y_pred'], predicciones['y_real'], predicciones['y_nwp']
        muestras = generar_muestras(**cfg.pmodel.resultados, y_pred=y_pred)
        for idx in tqdm(muestras):
            plt.plot(y_pred[idx,...].reshape(-1,1), 'red')
            plt.plot(y_real[idx,...].reshape(-1,1), 'green')
            plt.plot(y_nwp[idx,...].reshape(-1,1), 'magenta')
            plt.savefig(plots_path / f"{idx}.png", dpi=300)
            plt.close()
    if rain:
        print("Generando resultados de predicciones para precipitacion")
        cfg = AttrDict(parser(None, None, 'precipitacion')(copy.deepcopy(cfg_previo)))
        plots_path = Path(cfg.paths.pmodel.viz) 
        plots_path.mkdir(parents=True, exist_ok=True)
        print(f"\tUsando {cfg.paths.pmodel.predictions} como archivo de predicciones")
        with open(Path(cfg.paths.pmodel.predictions), 'rb') as handler:
            predicciones = pickle.load(handler)
        y_pred, y_real, y_nwp = predicciones['y_pred'], predicciones['y_real'], predicciones['y_nwp']
        muestras = generar_muestras(**cfg.pmodel.resultados, y_pred=y_pred)
        for idx in tqdm(muestras):
            plt.plot(np.argmax(y_real[idx,...], axis=-1).reshape(-1,1), 'green')
            plt.plot(np.argmax(y_pred[idx,...], axis=-1).reshape(-1,1), 'red')
            plt.plot(np.argmax(y_nwp[idx,...], axis=-1).reshape(-1,1), 'magenta')
            plt.savefig(plots_path / f"{idx}.png", dpi=300)
            plt.close()    
if __name__ == "__main__":
    main()