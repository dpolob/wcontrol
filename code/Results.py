# %%
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import click
import yaml
from pathlib import Path
from attrdict import AttrDict
from tqdm import tqdm

import common.utils.parser as parser

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
    cfg = parser.parser_experiment(cfg, name) # parser de {{experiment}}
    cfg = AttrDict(parser.parser_epoch(cfg, cfg.zmodel.dataloaders.test.use_checkpoint)) # parser de {{epoch}}
    with open(Path(cfg.paths.zmodel.predictions), 'rb') as handler:
        predicciones = pickle.load(handler)
        print("Cargado archivo de predicciones")
        
    if isinstance(cfg.zmodel.resultados.visualizacion.muestras, str):  # all, random o range
        if cfg.zmodel.resultados.visualizacion.muestras == 'range':  # range
            inicio = cfg.zmodel.resultados.visualizacion.inicio
            fin = len(predicciones) if cfg.zmodel.resultados.visualizacion.fin == 'end' else cfg.zmodel.resultados.visualizacion.fin
            paso = cfg.zmodel.resultados.visualizacion.paso
            muestras = range(inicio, fin, paso)
    
    plots_path = Path(cfg.paths.zmodel.viz) 
    plots_path.mkdir(parents=True, exist_ok=True)    
    for idx in tqdm(muestras):
        plt.plot(np.mean(predicciones.iloc[idx].loc['Y'], axis=0).reshape(-1,1), 'g')
        plt.plot(np.mean(predicciones.iloc[idx].loc['Ypred'], axis=0).reshape(-1,1), 'r')
        plt.savefig(plots_path / f"{idx}.png", dpi=300)
        plt.close()
        
        
if __name__ == "__main__":
    cli()