import click
import yaml
import pickle
import torch
import copy
import numpy as np
from datetime import datetime
from attrdict import AttrDict
from tqdm import tqdm
from colorama import Fore, Style
from pathlib import Path
from torch.utils.data import DataLoader

import common.predict.modules as predictor
from common.utils.parser import parser
import common.utils.kwargs_gen as generar_kwargs
from common.utils.datasets import dataset_pmodel as ds_pmodel
from common.utils.datasets import sampler_pmodel as sa_pmodel
from common.utils.datasets import dataset_pipeline as ds_pipeline
from common.utils.datasets import sampler_pipeline as sa_pipeline


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
    print(Fore.BLUE + "PREDICCION MODELO ZONAL" + Style.RESET_ALL)
    print(Fore.YELLOW + "Cargando archivos de configuracion y datos" + Style.RESET_ALL)

    try:
        print(f"\tArchivo de configuracion {file}", end="")
        with open(file, 'r') as handler:
            cfg = yaml.safe_load(handler)
            name = cfg["experiment"]
            epoch = cfg["zmodel"]["dataloaders"]["test"]["use_checkpoint"]
            cfg = AttrDict(parser(name, epoch)(cfg))
            print(f"\t\t\t\t\t\t{OK}")
        print(f"\tDatos de estaciones de zona {Path(cfg.paths.zmodel.dataset)}", end="")    
        with open(Path(cfg.paths.zmodel.dataset), 'rb') as handler:
            datasets = pickle.load(handler)
            print(f"\t\t\t\t{OK}")
        print(f"\tMetadatos de estaciones de zona {Path(cfg.paths.zmodel.dataset_metadata)}", end="")
        with open(Path(cfg.paths.zmodel.dataset_metadata), 'r') as handler:
            metadata = yaml.safe_load(handler)
            print(f"\t\t\t\t{OK}")
    except Exception as e:
        print(f"El archivo de configuracion del experimento no existe o no existe el archivo {cfg.paths.zmodel.dataset} \
            con los datasets para el modelo zonal o {cfg.paths.zmodel.dataset_metadata} de metadatos del dataset del \
            modelo zonal. Mas info: {e}")
        exit()
    
    if not cfg.zmodel.dataloaders.test.enable:
        print("El archivo no tiene definido dataset para test")
        exit()
    
    print(Fore.YELLOW + "Generando datos de prediccion " + Style.RESET_ALL)
    print(f"\tInicio del dataset en {metadata['fecha_min']}")
    kwargs_dataloader = generar_kwargs.dataloader(modelo='zmodel', fase='test', cfg=cfg, datasets=datasets, metadata=metadata)
    kwargs_prediccion = generar_kwargs.predict(modelo='zmodel', cfg=cfg)
 
    test_dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
    ### INVERTIR COMENTARIOS PARA DEPURACION 
    y_pred = predictor.predict(test_dataloader, **kwargs_prediccion)  # Si y_pred tiene N variables su shape es (len(test)), si no es (len(test), N, Ly, Fout)
    # with open('y_predzmodelpred.pickle', 'rb') as h:
    #    y_pred = pickle.load(h)

    y_real = np.empty_like(y_pred)
    y_nwp = np.empty_like(y_real)
    
    print(f"\tGenerando target para test", end='')
    for i, (_, _, _, Y, P) in enumerate(tqdm(test_dataloader, leave=False)):
        # hay que quitarles la componente 0 y pasarlos a numpy
        y_real[i,...] = Y[:, 1:, :].numpy()
        y_nwp[i, ...] = P[:, 1:, :].numpy()
    print(f"\t\t\t\t{OK}")
    predicciones = {'y_pred': y_pred, 'y_real': y_real, 'y_nwp': y_nwp}
   
    output = Path(cfg.paths.zmodel.predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\tUsando {output} como ruta para guardar predicciones")
    with open(output, 'wb') as handler:
        pickle.dump(predicciones, handler)
    print(f"\tSalvando archivo de predicciones en {output}\t\t{OK}")

@main.command()                                           
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
@click.option('--temp', is_flag=True, default= False, help='Predecir modelo de temperatura')
@click.option('--hr', is_flag=True, default= False, help='Predecir modelo de HR')
@click.option('--rain', is_flag=True, default= False, help='Predecir modelo de precipitacion')
def pmodel(file, temp, hr, rain):
    print(Fore.BLUE + "PREDICCION MODELO DE PARCELA" + Style.RESET_ALL)
    print(Fore.YELLOW + "Cargando archivos de configuracion y datos" + Style.RESET_ALL)
    try:
        print(f"\tArchivo de configuracion {file}", end="")
        with open(file, 'r') as handler:
            cfg = yaml.safe_load(handler)
            name = cfg["experiment"]
            epoch = cfg["pmodel"]["dataloaders"]["test"]["use_checkpoint"]
            cfg = AttrDict(parser(name, epoch)(cfg))
            print(f"\t\t\t\t\t\t{OK}")
        print(f"\tDatos de la estacion objetivo {Path(cfg.paths.pmodel.dataset)}", end="")    
        with open(Path(cfg.paths.pmodel.dataset), 'rb') as handler:
            datasets = pickle.load(handler)
            print(f"\t\t\t\t\t\t{OK}")
        print(f"\tMetadatos de la estacion objetivo {Path(cfg.paths.pmodel.dataset_metadata)}", end="")
        with open(Path(cfg.paths.pmodel.dataset_metadata), 'r') as handler:
            metadata = yaml.safe_load(handler)
            print(f"\t\t\t\t\t\t{OK}")
        print(f"\tMetadatos de estaciones de zona {Path(cfg.paths.zmodel.dataset_metadata)}", end="")
        with open(Path(cfg.paths.zmodel.dataset_metadata), 'r') as handler:
            metadata_zmodel = yaml.safe_load(handler)
            print(f"\t{OK}")
        print(f"\tDatos de estaciones de zona {Path(cfg.paths.zmodel.dataset)}", end="")    
        with open(Path(cfg.paths.zmodel.dataset), 'rb') as handler:
            dataset_zmodel = pickle.load(handler)
            print(f"\t{OK}")
    except Exception as e:
        print(f"El archivo de configuracion del experimento no existe o no existe el archivo {cfg.paths.pmodel.dataset} \
            con el dataset para el modelo de parcela o {cfg.paths.pmodel.dataset_metadata} de metadatos del dataset del \
            modelo de parcela. Mas info en: {e}")
        exit()
    
    
    device = 'cuda' if cfg.pmodel.model.use_cuda else 'cpu'
    Fout = len(cfg.prediccion)
    print(Fore.YELLOW + f"CUDA: {'SI' if device == 'cuda' else 'NO'}" + Style.RESET_ALL)
    
    ## Generar Dataset de test para acelerar el proceso de prediccion
    print(Fore.YELLOW + "Cargando datos temporales para acelerar el proceso de predicion" + Style.RESET_ALL, end='')
    if Path(cfg.paths.pmodel.pmodel_test).is_file():
        test_dataset = pickle.load(open(Path(cfg.paths.pmodel.pmodel_test), "rb"))
        print(f"\t\t\t\t\t\t{OK}")
    else:
        print(f"\t\t\t\t\t\t{FAIL}")
        print(Fore.YELLOW + "Generando datos de prediccion " + Style.RESET_ALL)
        Path(cfg.paths.pmodel.pmodel_test).parent.mkdir(parents=True, exist_ok=True)
        kwargs_dataloader = generar_kwargs.dataloader(modelo='pmodel', fase='test', cfg=cfg, datasets=dataset_zmodel, metadata=metadata_zmodel)
        dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
        kwargs_prediccion = generar_kwargs.predict(modelo='zmodel', cfg=cfg)
        y_pred_zmodel = predictor.predict(dataloader, **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
        print(f"\tRealizando conversion de {y_pred_zmodel.shape} a ", end='')
        # debemos tomar la media de las salidas
        y_t = np.empty(shape=(y_pred_zmodel.shape[0], cfg.futuro))
        y_hr =np.empty_like(y_t)
        y_rain=np.empty(shape=(y_pred_zmodel.shape[0], cfg.futuro, 8))
        if y_pred_zmodel.ndim == 1:  # el vector y_pred_zmodel tiene un shape de (len(test))
            for idx in range(y_pred_zmodel.shape[0]):
                y_t[idx, ...] = np.mean(y_pred_zmodel[idx][..., 0], axis=0)
                y_hr[idx, ...] = np.mean(y_pred_zmodel[idx][..., 1], axis=0)
                y_rain[idx, ...] = np.mean(y_pred_zmodel[idx][..., 2:], axis=0)
        else:  # el vector y_pred_zmodel tiene un shape de (len(test), N, Ly, Fout)        
            for idx in range(y_pred_zmodel.shape[0]):
                y_t[idx, ...] = np.mean(y_pred_zmodel[idx, ..., 0], axis=0)
                y_hr[idx, ...] = np.mean(y_pred_zmodel[idx, ..., 1], axis=0)
                y_rain[idx, ...] = np.mean(y_pred_zmodel[idx, ..., 2:], axis=0)
        
        y_t = np.expand_dims(y_t, axis=-1)
        y_hr = np.expand_dims(y_hr, axis=-1)
        for idx in range(y_rain.shape[0]):
            for idj in range(y_rain.shape[1]):
                y_rain[idx, idj] = np.where(y_rain[idx, idj] < max(y_rain[idx, idj]), 0, 1)
        y_rain.astype(int)
        y_pred_zmodel = np.concatenate([y_t,y_hr,y_rain], axis=-1)
    
        print(f"\t {y_pred_zmodel.shape}", end='')
        print(f"\t{OK}")
                
        print(f"\tGenerando target para test")
        y_real = np.empty((len(dataloader), cfg.futuro, Fout))
        kwargs_dataloader = generar_kwargs.dataloader(modelo='pmodel', fase='test', cfg=cfg, datasets=datasets, metadata=metadata)
        dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
        for i, (_, _, _, Y, _) in enumerate(tqdm(dataloader, leave=False)):
            # hay que quitarles la componente 0 y pasarlos a numpy porque son tensores
            y_real[i, ...] = Y[:, 1:, :].numpy()  # y_real = (len(dataset), Ly, Fout)
        print(f"\t{OK}")
        
        test_dataset = [y_pred_zmodel, y_real]
        print(f"\tGuardando datos en archivo temporal", end="")
        with open(Path(cfg.paths.pmodel.pmodel_test), 'wb') as handler:
            pickle.dump(test_dataset, handler)
        print(f"\t{OK}")
    
    
    
    cfg_previo = copy.deepcopy(dict(cfg))
    if not temp and not hr and not rain:
        print(f"No se ha definido que predecir. Ver pmodel --help {FAIL}")
        exit()
    if temp:
        ## Parte especifica temperatura
        print(Fore.YELLOW + "Prediccion modelo de temperatura..." + Style.RESET_ALL)
        cfg = AttrDict(parser(None, None, 'temperatura')(copy.deepcopy(cfg_previo)))
        test_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=test_dataset, componentes=slice(0, 1)),
                                        sampler=sa_pmodel.PModelSampler(datasets=test_dataset, batch_size=1, shuffle=False),    
                                        batch_size=None,
                                        num_workers=2)
       
        print(f"\tDataset de test: {len(test_dataloader)}", end='')
        kwargs_prediccion = generar_kwargs.predict(modelo='pmodel', cfg=cfg)
        y_pred = predictor.predict(test_dataloader, tipo='pmodel', **kwargs_prediccion)  # y_pred = (len(test), 1, Ly, Fout)
        y_pred = y_pred.squeeze(axis=1)  # (len(test), 1, Ly, Fout).squeeze(axis=1) -> (len(test), Ly, Fout)
        y_real = np.empty_like(y_pred)
        y_nwp = np.empty_like(y_pred)
        for i, (X, Y) in enumerate(tqdm(test_dataloader, leave=False)):
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
        print(Fore.YELLOW + "Prediccion modelo de humedad..." + Style.RESET_ALL)
        cfg = AttrDict(parser(None, None, 'hr')(copy.deepcopy(cfg_previo)))
        test_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=test_dataset, componentes=slice(1, 2)),
                                        sampler=sa_pmodel.PModelSampler(datasets=test_dataset, batch_size=1, shuffle=False),    
                                        batch_size=None,
                                        num_workers=2)
       
        print(f"\tDataset de test: {len(test_dataloader)}", end='')
        kwargs_prediccion = generar_kwargs.predict(modelo='pmodel', cfg=cfg) 
        y_pred = predictor.predict(test_dataloader, tipo='pmodel', **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
        y_pred = y_pred.squeeze(axis=1)  # (len(test), N, Ly, Fout).squeeze(axis=1) -> (len(test), Ly, Fout)
        y_real = np.empty_like(y_pred)
        y_nwp = np.empty_like(y_pred)
        for i, (X, Y) in enumerate(tqdm(test_dataloader, leave=False)):
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
        print(Fore.YELLOW + "Prediccion modelo de precipitacion..." + Style.RESET_ALL)
        cfg = AttrDict(parser(None, None, 'precipitacion')(copy.deepcopy(cfg_previo)))
        test_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=test_dataset, componentes=slice(2, 2 + len(metadata["bins"]))),
                                        sampler=sa_pmodel.PModelSampler(datasets=test_dataset, batch_size=1, shuffle=False),    
                                        batch_size=None,
                                        num_workers=2)
       
        print(f"\tDataset de test: {len(test_dataloader)}", end='')    
        kwargs_prediccion = generar_kwargs.predict(modelo='pmodel', cfg=cfg) 
        y_pred = predictor.predict(test_dataloader, tipo='pmodel', **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
        y_pred = y_pred.squeeze(axis=1)  # (len(test), N, Ly, Fout).squeeze(axis=1) -> (len(test), Ly, Fout)
        y_real = np.empty_like(y_pred)
        y_nwp = np.empty_like(y_pred)
        for i, (X, Y) in enumerate(tqdm(test_dataloader, leave=False)):
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

@main.command()                                           
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
def pipeline(file):
    print(Fore.BLUE + "PREDICCION PIPELINE" + Style.RESET_ALL)
    print(Fore.YELLOW + "Cargando archivos de configuracion y datos" + Style.RESET_ALL)
    try:
        print(f"\tArchivo de configuracion {file}", end="")
        with open(file, 'r') as handler:
            cfg = yaml.safe_load(handler)
            name = cfg["experiment"]
            cfg = AttrDict(parser(name, None)(cfg))
            print(f"\t\t\t\t\t\t{OK}")
        print(f"\tDatos de estaciones de zona {Path(cfg.paths.pipeline.dataset_zona)}", end="")
        with open(Path(cfg.paths.pipeline.dataset_zona), 'rb') as handler:
            datasets_zona = pickle.load(handler)
            print(f"\t\t{OK}")
        print(f"\tDatos de la estacion objetivo {Path(cfg.paths.pipeline.dataset)}", end="")
        with open(Path(cfg.paths.pipeline.dataset), 'rb') as handler:
            datasets_pmodel = pickle.load(handler)
            print(f"\t{OK}")
        print(f"\tMetadatos de estaciones de zona {Path(cfg.paths.pipeline.dataset_zona_metadata)}", end="")
        with open(Path(cfg.paths.pipeline.dataset_zona_metadata), 'r') as handler:
            metadata_zona = yaml.safe_load(handler)
            print(f"\t{OK}")
        print(f"\tMetadatos de la estacion objetivo {Path(cfg.paths.pipeline.dataset_metadata)}", end="")
        with open(Path(cfg.paths.pipeline.dataset_metadata), 'r') as handler:
            metadata_pmodel = yaml.safe_load(handler)
            print(f"\t{OK}")
    except Exception as e:
        print(f"\t{FAIL}")
        print(f"Algunos de los arcivos no existe. Mas info en: {e}")
        exit()
    
    print(Fore.YELLOW + "Prediccion con el modelo zonal del conjunto de test desde datos del dataset de estaciones zonal"+ Style.RESET_ALL)
    # Realizar un predict del zmodel con los datos de la zona, guardados en dataset_zmodel con las fechas de test de pmodel
    kwargs_dataloader = generar_kwargs.dataloader(modelo='zmodel', fase='test', cfg=cfg, datasets=datasets_zona, metadata=metadata_zona)
    kwargs_prediccion = generar_kwargs.predict(modelo='zmodel', cfg=cfg)
    test_dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
    
    # el if es solo para acelerar el proceso...
    temporal = Path(cfg.paths.pipeline.temp) / 'a.pickle'
    if temporal.is_file():
        print(f"\tCargando archivo temporal de predicciones", end='')
        y_pred_zmodel = pickle.load(open(temporal, 'rb'))
        print(f"\t\t\t\t\t{OK}")
    else:
        print(f"\tCalculando predicciones")
        y_pred_zmodel = predictor.predict(test_dataloader, **kwargs_prediccion)  # y_pred puede ser (len(test)) o (len(test), N, Ly, Fout)
        temporal.parent.mkdir(parents=True, exist_ok=True)
        print(f"\tGuardando predicciones en archivo temporal", end="")
        with open(Path(cfg.paths.pipeline.temp) / 'a.pickle', 'wb') as handler:
            pickle.dump(y_pred_zmodel, handler)
            print(f"\t{OK}")
   
   
   
    print(f"\tRealizando conversion de {y_pred_zmodel.shape} a ", end='')
    # debemos tomar la media de las salidas
    y_t = np.empty(shape=(y_pred_zmodel.shape[0], cfg.futuro))
    y_hr =np.empty_like(y_t)
    y_rain=np.empty(shape=(y_pred_zmodel.shape[0], cfg.futuro, 8))
    if y_pred_zmodel.ndim == 1:  # el vector y_pred_zmodel tiene un shape de (len(test))
        for idx in range(y_pred_zmodel.shape[0]):
            y_t[idx, ...] = np.mean(y_pred_zmodel[idx][..., 0], axis=0)
            y_hr[idx, ...] = np.mean(y_pred_zmodel[idx][..., 1], axis=0)
            y_rain[idx, ...] = np.mean(y_pred_zmodel[idx][..., 2:], axis=0)
    else:  # el vector y_pred_zmodel tiene un shape de (len(test), N, Ly, Fout)        
        for idx in range(y_pred_zmodel.shape[0]):
            y_t[idx, ...] = np.mean(y_pred_zmodel[idx, ..., 0], axis=0)
            y_hr[idx, ...] = np.mean(y_pred_zmodel[idx, ..., 1], axis=0)
            y_rain[idx, ...] = np.mean(y_pred_zmodel[idx, ..., 2:], axis=0)
    y_t = np.expand_dims(y_t, axis=-1)
    y_hr = np.expand_dims(y_hr, axis=-1)
    for idx in range(y_rain.shape[0]):
        for idj in range(y_rain.shape[1]):
            y_rain[idx, idj] = np.where(y_rain[idx, idj] < max(y_rain[idx, idj]), 0, 1)
    
    y_pred_zmodel = np.concatenate([y_t,y_hr,y_rain], axis=-1)
    print(f"\t {y_pred_zmodel.shape}", end='')
    print(f"\t{OK}")

    print(Fore.YELLOW + "Obteniendo valores reales y de prevision del conjunto de test desde el dataset de la estacion objetivo" + Style.RESET_ALL)
    # tambien necesitamos el y_real y y_nwp que estan en dataset_pmodel 
    kwargs_dataloader = generar_kwargs.dataloader(modelo='pmodel', fase='test', cfg=cfg, datasets=datasets_pmodel, metadata=metadata_pmodel)
    test_dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
    y_real = np.empty(shape=(len(test_dataloader), cfg.futuro, len(cfg.prediccion)))
    y_nwp = np.empty(shape=(len(test_dataloader), cfg.futuro, len(cfg.prediccion)))
    # el if es solo para acelerar el proceso...
    temporal = Path(cfg.paths.pipeline.temp) / 'b.pickle'
    if temporal.is_file():
        print(f"\tCargando archivo temporal de valores reales y de prevision...", end="")
        y_real, y_nwp = pickle.load(open(temporal, 'rb'))
        print(f"\t{OK}")
    else:
        print("\tGenerando datos")
        for i, (_, _, _, Y, P) in enumerate(tqdm(test_dataloader, leave=False)):  # P.shape = Y.shape  = (len(test), Ly + 1, Fout)
            y_real[i, ...] = Y[:, 1:, :].numpy()      # hay que quitarles la componente 0 y pasarlos a numpy
            y_nwp[i, ...] = P[:, 1:, :].numpy()      # hay que quitarles la componente 0 y pasarlos a numpy
        temporal.parent.mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.paths.pipeline.temp) / 'b.pickle', 'wb') as handler:
            pickle.dump([y_real, y_nwp], handler)
    
    # print(f"{y_pred_zmodel.shape=}")
    # print(f"{y_real.shape=}")
    # print(f"{y_nwp.shape=}")

    # Realizar un predict del pmodel con los datos de la estacion pmodel
    # para cada uno de los modelos definidos
    
    def generar(modelo: str) -> np.ndarray:
        if modelo == 'temperatura':
            cfg_inner = AttrDict(parser(None, None, 'temperatura')(copy.deepcopy(cfg_previo)))
            componente = slice(0, 1)
        elif modelo == 'hr':
            cfg_inner = AttrDict(parser(None, None, 'hr')(copy.deepcopy(cfg_previo)))
            componente = slice(1, 2)
        elif modelo == 'precipitacion':
            cfg_inner = AttrDict(parser(None, None, 'precipitacion')(copy.deepcopy(cfg_previo)))
            componente = slice(2, 2 + len(metadata_pmodel['bins']))
        else:
            raise NotImplementedError  
            
        pipeline_dataloader = DataLoader(dataset=ds_pipeline.PipelineDataset(datasets=y_pred_zmodel, componentes=componente),
                                        sampler=sa_pipeline.PipelineSampler(datasets=y_pred_zmodel, batch_size=1, shuffle=False),    
                                        batch_size=None,
                                        num_workers=2)
        kwargs_prediccion = generar_kwargs.predict(modelo='pmodel', cfg=cfg_inner)
        y_pred = predictor.predict(pipeline_dataloader, tipo='pmodel', **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
        return y_pred.squeeze(axis=1)  # (len(test), N, Ly, Fout).squeeze(axis=1) -> (len(test), Ly, Fout)
        
    
    cfg_previo = copy.deepcopy(dict(cfg))
    print(Fore.YELLOW + "Prediccion modelo de temperatura" + Style.RESET_ALL)
    y_pred_temp = generar('temperatura')
    print(Fore.YELLOW + "Prediccion modelo de hr" + Style.RESET_ALL)
    y_pred_hr = generar('hr')
    print(Fore.YELLOW + "Prediccion modelo de precipitacion" + Style.RESET_ALL)
    y_pred_rain = generar('precipitacion')
       
    predicciones = {'y_real': y_real, 'y_pred_hr': y_pred_hr, 'y_pred_rain': y_pred_rain, 'y_pred_temp': y_pred_temp, 'y_nwp': y_nwp}    
   
    output = Path(cfg.paths.pipeline.predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(Fore.YELLOW + f"Guardando archivo de predicciones en {output}" + Style.RESET_ALL, end="")
    with open(output, 'wb') as handler:
        pickle.dump(predicciones, handler)
        print(f"\t\t\t\t\t{OK}")
   
    

if __name__ == "__main__":
    main()