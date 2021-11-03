import click
import yaml
import pickle

from pathlib import Path
from attrdict import AttrDict
from colorama import Fore, Style

import common.utils.parser as parser
#from common.data_preprocessing.modules import generarvariablesZmodel, generarvariablesPmodel
from common.data_preprocessing.modules import generarvariablesZmodel, generarvariablesPmodel

OK = "\t[ " + Fore.GREEN +"OK" + Style.RESET_ALL + " ]"
FAIL = "\t[ " + Fore.RED + "FAIL" + Style.RESET_ALL + " ]"


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
    cfg = AttrDict(parser.parser_experiment(cfg, name)) # parser de {{experiment}}
    dfs, metadata = generarvariablesZmodel(estaciones=list(cfg.zmodel.estaciones), 
                                           escaladores = cfg.preprocesado.escaladores, 
                                           outliers = cfg.preprocesado.outliers,
                                           proveedor= cfg.zmodel.proveedor[0])
    output = Path(cfg.paths.zmodel.dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Guardando salida en {cfg.paths.zmodel.dataset}", end='')
    with open(output, 'wb') as handler:
        pickle.dump(dfs, handler)
    print(OK)         
    print(f"Guardando metadatos en {cfg.paths.zmodel.dataset_metadata}", end='')                    
    with open(Path(cfg.paths.zmodel.dataset_metadata), 'w') as handler:
        yaml.safe_dump(metadata, handler, allow_unicode=True)
    print(OK)
    
@cli.command()
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
def pmodel(file):
    try:
        with open(file, 'r') as handler:
            cfg = yaml.safe_load(handler)
        print(f"Usando {file} como archivo de configuracion")
    except:
        print(f"{file} no existe. Por favor defina un archivo con --file")
        exit()
    name = cfg["experiment"]
    cfg = AttrDict(parser.parser_experiment(cfg, name))
    output = Path(cfg.paths.pmodel.dataset)
    output.parent.mkdir(parents=True, exist_ok=True)

    estacion = list(cfg.pmodel.estaciones)[0]
    escaladores = cfg.preprocesado.escaladores
    outliers = cfg.preprocesado.outliers
    
    df, metadata = generarvariablesPmodel(estacion=list(cfg.pmodel.estaciones)[0], 
                                estaciones=cfg.zmodel.estaciones, escaladores=cfg.preprocesado.escaladores,
                                outliers= cfg.preprocesado.outliers, cfg=cfg)
    print(f"Guardando salida en {cfg.paths.pmodel.dataset}", end='')
    
    with open(output, 'wb') as handler:
        pickle.safe_dump(df, handler)
    print(OK)

    
if __name__ == "__main__":
    cli()
