import click
import yaml
import pickle

from pathlib import Path
from attrdict import AttrDict
from colorama import Fore, Style

from common.utils.parser import parser
from common.data_preprocessing.modules import generar_variables

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
        print(f"Usando {file} como archivo de configuracion")
    except:
        print(f"{file} no existe. Por favor defina un archivo con --file {FAIL}")
        exit()
    
    name = cfg["experiment"]
    cfg = AttrDict(parser(name, None)(cfg))
    dfs, metadata = generar_variables(estaciones=list(cfg.zmodel.estaciones), 
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
    
@main.command()
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
def pmodel(file):
    try:
        with open(file, 'r') as handler:
            cfg = yaml.safe_load(handler)
            name = cfg["experiment"]
            cfg = AttrDict(parser(name, None)(cfg))
        print(f"Usando {file} como archivo de configuracion")
        with open(Path(cfg.paths.zmodel.dataset_metadata), 'r') as handler:
            dataset_metadata = yaml.safe_load(handler)
    except:
        print(f"El archivo de configuracion del experimento o el de metadatos del modelo zonal no existe. {FAIL}")
        exit()
    
    name = cfg["experiment"]
    cfg = AttrDict(parser(name, None)(cfg))
    output = Path(cfg.paths.pmodel.dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    dfs, metadata = generar_variables(estaciones=list(cfg.pmodel.estaciones), 
                                      outliers = cfg.preprocesado.outliers,
                                      proveedor= cfg.zmodel.proveedor[0],
                                      CdG=list(dataset_metadata["CdG"]))
    print(f"Guardando salida en {cfg.paths.pmodel.dataset}", end='')
    
    with open(output, 'wb') as handler:
        pickle.dump(dfs, handler)
    print(OK)
    print(f"Guardando metadatos en {cfg.paths.pmodel.dataset_metadata}", end='')                    
    with open(Path(cfg.paths.pmodel.dataset_metadata), 'w') as handler:
        yaml.safe_dump(metadata, handler, allow_unicode=True)
    print(OK)

if __name__ == "__main__":
    main()
