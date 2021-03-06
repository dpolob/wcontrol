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
import common.utils.kwargs_gen as generar_kwargs
from common.utils.datasets import dataset_seq2seq as ds
from common.utils.datasets import dataset_pmodel as ds_pmodel
from common.utils.datasets import sampler_seq2seq as sa
from common.utils.datasets import sampler_pmodel as sa_pmodel
from common.utils.trainers import trainerSeq2Seq as tr
from common.utils.trainers import trainerpmodel as tr_pmodel
from common.utils.loss_functions import lossfunction as lf
import models.pmodel.p_model as md_pmodel
import importlib

torch.manual_seed(420)
np.random.seed(420)   
SI = Fore.GREEN + "SI" + Style.RESET_ALL
NO = Fore.RED + "NO" + Style.RESET_ALL

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
            cfg = AttrDict(parser(name, None)(cfg))
        print(f"Usando {file} como archivo de configuracion")
        with open(Path(cfg.paths.zmodel.dataset_metadata), 'r') as handler:
            metadata = yaml.safe_load(handler)
        print("Leidos metadatos del dataset")
        with open(Path(cfg.paths.zmodel.dataset), 'rb') as handler:
            datasets = pickle.load(handler)
        print(f"Leidos metadatos del dataset")
    except Exception as e:
        print(f"El archivo de configuracion del experimento no existe o no existe el archivo {cfg.paths.zmodel.dataset} \
            con el dataset para el modelo zonal o {cfg.paths.zmodel.dataset_metadata} de metadatos del dataset del \
            modelo zonal. Mas info en: {e}")
        exit()
    
    runs_path = Path(cfg.paths.zmodel.runs)
    runs_path.mkdir(parents=True, exist_ok=True)
    print(f"Usando {runs_path} como ruta para runs")
    chkpts_path = Path(cfg.paths.zmodel.checkpoints)
    chkpts_path.mkdir(parents=True, exist_ok=True)
    print(f"Usando {chkpts_path} como ruta para guardar checkpoints")
         
    device = 'cuda' if cfg.zmodel.model.use_cuda else 'cpu'
    PASADO = cfg.pasado
    FUTURO = cfg.futuro
    EPOCHS = cfg.zmodel.model.epochs
    kwargs_loss = cfg.zmodel.model.loss_function
    Fout = list(cfg.prediccion)  # etiquetas a predecir len = Fout
    Ff = list(cfg.zmodel.model.encoder.features)  # etiquetas para encoder len = Ff
    Ft = list(cfg.zmodel.model.decoder.features)  # etiquetas para decoder len = Ft
    Fnwp = list(cfg.zmodel.model.decoder.nwp)  # etiquetas de los modelos nwp len = Fnwp
     
    TRAIN = cfg.zmodel.dataloaders.train.enable
    if TRAIN:
        FECHA_INICIO_TRAIN = datetime.strptime(cfg.zmodel.dataloaders.train.fecha_inicio, "%Y-%m-%d %H:%M:%S")
        FECHA_FIN_TRAIN = datetime.strptime(cfg.zmodel.dataloaders.train.fecha_fin, "%Y-%m-%d %H:%M:%S")
    else:
        FECHA_INICIO_TRAIN = None
        FECHA_FIN_TRAIN = None
    
    VALIDATION = cfg.zmodel.dataloaders.validation.enable
    if VALIDATION:
        FECHA_INICIO_VALID = datetime.strptime(cfg.zmodel.dataloaders.validation.fecha_inicio, "%Y-%m-%d %H:%M:%S")
        FECHA_FIN_VALID = datetime.strptime(cfg.zmodel.dataloaders.validation.fecha_fin, "%Y-%m-%d %H:%M:%S")
    else:
        FECHA_INICIO_VALID = None
        FECHA_FIN_VALID = None
    
    print(f"Train: {SI if TRAIN else NO}, Validation: {SI if VALIDATION else NO}\n")
    
    fecha_inicio = datetime.strptime(metadata['fecha_min'], "%Y-%m-%d %H:%M:%S")
       
    # Split datasets
    if TRAIN:
        x = (FECHA_INICIO_TRAIN - fecha_inicio).days * 24 + (FECHA_INICIO_TRAIN - fecha_inicio).seconds / 3600
        y = (FECHA_FIN_TRAIN - fecha_inicio).days * 24 + (FECHA_FIN_TRAIN - fecha_inicio).seconds / 3600
        # x=0; y=500
        print(f"Generando dataset de train desde {x} a {y}")
        dfs_train = [df.loc[(df.index >= x ) & (df.index <= y), :] for df in datasets]
    if VALIDATION:
        x = (FECHA_INICIO_VALID - fecha_inicio).days * 24 + (FECHA_INICIO_VALID - fecha_inicio).seconds / 3600
        y = (FECHA_FIN_VALID - fecha_inicio).days * 24 + (FECHA_FIN_VALID - fecha_inicio).seconds / 3600
        # x=1001; y=1500
        print(f"Generando dataset de validation desde {x} a {y}")
        dfs_valid = [df.loc[(df.index >= x ) & (df.index <= y), :] for df in datasets]
   
    shuffle = True if 'shuffle' not in cfg.zmodel.dataloaders.train.keys() else cfg.zmodel.dataloaders.train.shuffle
    if TRAIN:
        train_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(datasets=dfs_train,
                                                                pasado=PASADO,
                                                                futuro=FUTURO,
                                                                etiquetaX=Fout,
                                                                etiquetaF=Ff,
                                                                etiquetaT=Ft,
                                                                etiquetaP=Fnwp),
                                      sampler=sa.Seq2SeqSampler(datasets=dfs_train,
                                                                pasado=PASADO,
                                                                futuro=FUTURO,
                                                                shuffle=shuffle),
    
                                      batch_size=None,
                                      num_workers=8)
    shuffle = True if 'shuffle' not in cfg.zmodel.dataloaders.validation.keys() else cfg.zmodel.dataloaders.validation.shuffle
    if VALIDATION:
        valid_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(datasets=dfs_valid,
                                                                pasado=PASADO,
                                                                futuro=FUTURO,
                                                                etiquetaX=Fout,
                                                                etiquetaF=Ff,
                                                                etiquetaT=Ft,
                                                                etiquetaP=Fnwp),
                                      sampler=sa.Seq2SeqSampler(datasets=dfs_valid,
                                                                pasado=PASADO,
                                                                futuro=FUTURO,
                                                                shuffle=shuffle),
    
                                      batch_size=None,
                                      num_workers=8)
   
    print(f"Dataset de train: {len(dfs_train)} componentes con longitud m??xima de {len(train_dataloader)}")
    print(f"Dataset de validacion: {len(dfs_valid)} componentes con longitud m??xima de {len(valid_dataloader)}")
   
    module = importlib.import_module(f"models.seq2seq.{cfg.zmodel.model.name}")
    encoder = module.RNNEncoder(rnn_num_layers=cfg.zmodel.model.encoder.rnn_num_layers,
                                input_feature_len=len(Ff),
                                sequence_len=PASADO + 1, # +1 porque tomamos tiempo0
                                hidden_size=cfg.zmodel.model.encoder.hidden_size,
                                bidirectional=cfg.zmodel.model.encoder.bidirectional,
                                device=device,
                                rnn_dropout=cfg.zmodel.model.encoder.rnn_dropout)
    encoder = encoder.to(device)

    decoder = module.DecoderCell(input_feature_len=len(Ft) + len(Fnwp),
                                 hidden_size=cfg.zmodel.model.decoder.hidden_size,
                                 output_size=len(Fout),
                                 dropout=cfg.zmodel.model.decoder.dropout,
                                 regres_hidden_layers=dict(cfg.zmodel.model.decoder.temphr_n_layers),
                                 class_hidden_layers=dict(cfg.zmodel.model.decoder.class_n_layers),
                                 bins_len=len(metadata['bins']))
    decoder = decoder.to(device)

    model = module.EncoderDecoderWrapper(encoder=encoder,
                                    decoder_cell=decoder,
                                    output_size=len(Fout),
                                    output_sequence_len=FUTURO,
                                    device=device)
    model = model.to(device)

    # Funciones loss
    loss_fn = lf.LossFunction(**kwargs_loss)
    encoder_optimizer = getattr(optim, cfg.zmodel.model.encoder.optimizer_name)(encoder.parameters(), lr=cfg.zmodel.model.encoder.lr, weight_decay=cfg.zmodel.model.decoder.lr / 10)
    decoder_optimizer = getattr(optim, cfg.zmodel.model.decoder.optimizer_name)(decoder.parameters(), lr=cfg.zmodel.model.decoder.lr, weight_decay=cfg.zmodel.model.decoder.lr / 10)
       
    trainer = tr.TorchTrainer(name=name,
                              model=model,
                              optimizer=[encoder_optimizer, decoder_optimizer], 
                              loss_fn = loss_fn,   
                              scheduler=None,
                              device=device,
                              # scheduler_batch_step=True if cfg.zmodel.model.scheduler else False,
                              checkpoint_folder= chkpts_path,
                              runs_folder= runs_path,
                              save_model=cfg.zmodel.model.save_model,
                              save_model_path=cfg.paths.zmodel.model,
                              early_stop=cfg.zmodel.model.early_stop,
                              alpha = cfg.zmodel.model.decoder.alpha,
                              keep_best_checkpoint=False
                              )
    if TRAIN and VALIDATION:
        trainer.train(EPOCHS, train_dataloader, valid_dataloader, resume_only_model=True, resume=True, plot=cfg.zmodel.model.plot_intermediate_results)
    else:
        trainer.train(EPOCHS, train_dataloader  , resume_only_model=True, resume=True, plot=cfg.zmodel.model.plot_intermediate_results)

    with open(chkpts_path / "valid_losses.pickle", 'rb') as handler:
        valid_losses = pickle.load(handler)
    if valid_losses != {}:
        best_epoch = sorted(valid_losses.items(), key=lambda x:x[1])[0][0]
        print(f"Mejor loss de validacion: {best_epoch}")


@main.command()
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
@click.option('--temp', is_flag=True, default= False, help='Entrenar modelo de temperatura')
@click.option('--hr', is_flag=True, default= False, help='Entrenar modelo de HR')
@click.option('--rain', is_flag=True, default= False, help='Entrenar modelo de precipitacion')
def pmodel(file, temp, hr, rain):
    print(Fore.YELLOW + "Cargando archivos de configuracion y datos" + Style.RESET_ALL)
    try:
        print(f"\tArchivo de configuracion {file}", end="")
        with open(file, 'r') as handler:
            cfg = yaml.safe_load(handler)
            name = cfg["experiment"]
            cfg = AttrDict(parser(name, None)(cfg))
            print(f"\t\t\t\t\t\t{OK}")
        print(f"\tMetadatos de la estacion objetivo {Path(cfg.paths.pmodel.dataset_metadata)}", end="")
        with open(Path(cfg.paths.pmodel.dataset_metadata), 'r') as handler:
            metadata = yaml.safe_load(handler)
            print(f"\t{OK}")
        print(f"\Datos de la estacion objetivo {Path(cfg.paths.pmodel.dataset)}", end="")    
        with open(Path(cfg.paths.pmodel.dataset), 'rb') as handler:
            datasets = pickle.load(handler)
            print(f"\t{OK}")
        print(f"\tMetadatos de estaciones de zona {Path(cfg.paths.zmodel.dataset_metadata)}", end="")
        with open(Path(cfg.paths.zmodel.dataset_metadata), 'r') as handler:
            metadata_zmodel = yaml.safe_load(handler)
            print(f"\t{OK}")
        print(f"\tDatos de estaciones de zona {Path(cfg.paths.zmodel.dataset)}", end="")    
        with open(Path(cfg.paths.zmodel.dataset), 'rb') as handler:
            dataset_zmodel = pickle.load(handler)
            print(f"Usando {cfg.paths.zmodel.dataset} como archivo de datos procesados de estaciones")
    except Exception as e:
        print(f"Alguno de los archivos no existe. Mas info en: {e}")
        exit()
        
    device = 'cuda' if cfg.pmodel.model.use_cuda else 'cpu'
    print(Fore.YELLOW + f"CUDA: {SI if device == 'cuda' else NO}"+ Style.RESET_ALL)
    Fout = len(cfg.prediccion)
    
    ## Generar Dataset de train
    print(Fore.YELLOW + "Prediccion con el modelo zonal del conjunto de train del dataset de estaciones zonal"+ Style.RESET_ALL)
    
    print(f"\tCargando datos de train desde el archivo temporarl existente {cfg.paths.pmodel.pmodel_train}", end='')
    if Path(cfg.paths.pmodel.pmodel_train).is_file():
        train_dataset = pickle.load(open(Path(cfg.paths.pmodel.pmodel_train), "rb"))
        print(f"\t{OK}")
    else:
        print(f"{FAIL}")
        print(f"\tGenerando datos de train")
        Path(cfg.paths.pmodel.pmodel_train).parent.mkdir(parents=True, exist_ok=True)
        kwargs_dataloader = generar_kwargs.dataloader(modelo='pmodel', fase='train', cfg=cfg, datasets=dataset_zmodel, metadata=metadata_zmodel)
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
                
        print(f"\tGenerando target para train")
        y_real = np.empty((len(dataloader), cfg.futuro, Fout))
        kwargs_dataloader = generar_kwargs.dataloader(modelo='pmodel', fase='train', cfg=cfg, datasets=datasets, metadata=metadata)
        dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
        for i, (_, _, _, Y, _) in enumerate(tqdm(dataloader, leave=False)):
            # hay que quitarles la componente 0 y pasarlos a numpy porque son tensores
            y_real[i, ...] = Y[:, 1:, :].numpy()  # y_real = (len(dataset), Ly, Fout)
        print(f"\t{OK}")
        
        train_dataset = [y_pred_zmodel, y_real]
        print(f"\tGuardando datos en archivo temporal", end="")
        with open(Path(cfg.paths.pmodel.pmodel_train), 'wb') as handler:
            pickle.dump(train_dataset, handler)
        print(f"\t{OK}")

    ## Generar Dataset de validacion
    print(Fore.YELLOW + "Prediccion con el modelo zonal del conjunto de validacion del dataset de estaciones zonal"+ Style.RESET_ALL)
    print(f"Cargango datos de validacion desde el modelo zmodel desde {cfg.paths.pmodel.pmodel_valid}", end='')
    if Path(cfg.paths.pmodel.pmodel_valid).is_file():
        valid_dataset = pickle.load(open(Path(cfg.paths.pmodel.pmodel_valid), "rb"))
        print(f"\t{OK}")
    else:
        print(f"{FAIL}")
        print(f"Generando datos de validacion")
        Path(cfg.paths.pmodel.pmodel_valid).parent.mkdir(parents=True, exist_ok=True)
        kwargs_dataloader = generar_kwargs.dataloader(modelo='pmodel', fase='validation', cfg=cfg, datasets=dataset_zmodel, metadata=metadata_zmodel)
        dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
        kwargs_prediccion = generar_kwargs.predict(modelo='zmodel', cfg=cfg)
        y_pred_zmodel = predictor.predict(dataloader, **kwargs_prediccion)  # y_pred = (len(test)) que continen (N, Ly, Fout)
        # dependiendo si N es variable devuelve un objeto (len(test)) que contiene arrays (N, Ly, Fout) o un array (len(test), N, Ly, Fout))
        
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
    
        print(f"\tGenerando target para train")
        y_real = np.empty((len(dataloader), cfg.futuro, Fout))
        kwargs_dataloader = generar_kwargs.dataloader(modelo='pmodel', fase='validation', cfg=cfg, datasets=datasets, metadata=metadata)
        dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
 
        for i, (_, _, _, Y, _) in enumerate(tqdm(dataloader, leave=False)):
            # hay que quitarles la componente 0 y pasarlos a numpy porque son tensores
            y_real[i, ...] = Y[:, 1:, :].numpy()  # y_real = (len(dataset), Ly, Fout)
        print(f"\t{OK}")
        valid_dataset = [y_pred_zmodel, y_real]
        print(f"\tGuardando datos en archivo temporal", end="")
        with open(Path(cfg.paths.pmodel.pmodel_valid), 'wb') as handler:
            pickle.dump(valid_dataset, handler)
        print(f"\t{OK}")

    def generar_modelo(modelo: str, ) -> None:
        if modelo == 'temperatura':
            cfg_inner = AttrDict(parser(None, None, 'temperatura')(copy.deepcopy(cfg_previo)))
            handler = cfg_inner.pmodel.model.temperatura
            F = 1
            componente = slice(0, 1)
        elif modelo == 'hr':
            cfg_inner = AttrDict(parser(None, None, 'hr')(copy.deepcopy(cfg_previo)))
            handler = cfg_inner.pmodel.model.hr
            F = 1
            componente = slice(1, 2)
        elif modelo == 'precipitacion':
            cfg_inner = AttrDict(parser(None, None, 'precipitacion')(copy.deepcopy(cfg_previo)))
            handler = cfg_inner.pmodel.model.precipitacion
            F = len(metadata['bins'])
            componente = slice(2, 2 + len(metadata['bins']))
        else:
            raise NotImplementedError  
        
        EPOCHS = handler.epochs
        TRAIN = cfg_inner.pmodel.dataloaders.train.enable
        VALIDATION = cfg_inner.pmodel.dataloaders.validation.enable
        print(Fore.YELLOW + f"Usando {cfg_inner.paths.pmodel.runs} como ruta para runs" + Style.RESET_ALL)
        print(Fore.YELLOW + f"Usando {cfg_inner.paths.pmodel.checkpoints} como ruta para guardar checkpoints" + Style.RESET_ALL)
        print(f"Train: {SI if TRAIN else NO}, Validation: {SI if VALIDATION else NO}\n")
        shuffle = False if 'shuffle' not in cfg_inner.pmodel.dataloaders.train.keys() else cfg_inner.pmodel.dataloaders.train.shuffle
        if TRAIN:
            train_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=train_dataset, componentes=componente),
                                        sampler=sa_pmodel.PModelSampler(datasets=train_dataset, batch_size=1, shuffle=shuffle),    
                                        batch_size=None,
                                        num_workers=2)
        shuffle = False if 'shuffle' not in cfg_inner.pmodel.dataloaders.validation.keys() else cfg_inner.pmodel.dataloaders.validation.shuffle
        if VALIDATION:
            valid_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=valid_dataset, componentes=componente),    
                                        sampler=sa_pmodel.PModelSampler(datasets=valid_dataset, batch_size=1, shuffle=shuffle),
                                        batch_size=None,
                                        num_workers=2)
            
        print(f"\tDataset de train: {len(train_dataloader)}")
        print(f"\tDataset de validacion: {len(valid_dataloader)}")
        kwargs_model={"Fin": F, 
                      "Fout": F,
                      "gru_num_layers": handler.gru_n_layers,
                      "hidden_size": handler.hidden_size,
                      "hidden_layers": handler.hidden_layers}
        model = md_pmodel.RedGeneral(**kwargs_model)
        model = model.to(device)
        ## Funciones loss
        kwargs_loss = handler.loss_function
        loss_fn = lf.LossFunction(**kwargs_loss)
        model_optimizer = getattr(optim, handler.optimizer_name)(model.parameters(), lr=handler.lr, weight_decay=handler.lr / 10)
        trainer = tr_pmodel.TorchTrainer( model=model, optimizer=model_optimizer, loss_fn = loss_fn, device=device,
                                         checkpoint_folder= Path(cfg_inner.paths.pmodel.checkpoints),
                                         runs_folder= Path(cfg_inner.paths.pmodel.runs),
                                         keep_best_checkpoint=True,
                                         save_model= handler.save_model,
                                         save_model_path=Path(cfg_inner.paths.pmodel.model),
                                         early_stop=handler.early_stop)
        if TRAIN and VALIDATION:
            trainer.train(EPOCHS, train_dataloader, valid_dataloader, resume_only_model=True, resume=True, plot=handler.plot_intermediate_results, rain=True if modelo == 'precipitacion' else False)
        else:
            trainer.train(EPOCHS, train_dataloader, resume_only_model=True, resume=True, plot=handler.plot_intermediate_results, rain=True if modelo == 'precipitacion' else False)
        
        with open(Path(cfg_inner.paths.pmodel.checkpoints) / "valid_losses.pickle", 'rb') as handler:
            valid_losses = pickle.load(handler)
        if valid_losses != {}:
            best_epoch = sorted(valid_losses.items(), key=lambda x:x[1])[0][0]
            print(Fore.GREEN + f"Mejor loss de validacion: {best_epoch}" + Style.RESET_ALL)
    
    cfg_previo = copy.deepcopy(dict(cfg))
    if not temp and not hr and not rain:
        print(Fore.RED + "No se ha definido modelo para entrenar" + Style.RESET_ALL)
        exit()
        
    if temp:
        ## Parte especifica temperatura
        print(Fore.YELLOW + "Entrenado modelo de temperatura..." + Style.RESET_ALL)
        generar_modelo('temperatura')
    if hr:
        print(Fore.YELLOW + "Entrando modelo de hr..."+ Style.RESET_ALL)
        generar_modelo('hr')
    if rain:
        print(Fore.YELLOW + "Entrando modelo de precipitacion..."+ Style.RESET_ALL)
        generar_modelo('precipitacion')


if __name__ == "__main__":
    main()
