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

torch.manual_seed(420)
np.random.seed(420)   
SI = Fore.GREEN + "SI" + Style.RESET_ALL
NO = Fore.RED + "NO" + Style.RESET_ALL


    

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
        print(f"Usando {name} como nombre del experimento")
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
   
    print(f"Dataset de train: {len(dfs_train)} componentes con longitud máxima de {len(train_dataloader)}")
    print(f"Dataset de validacion: {len(dfs_valid)} componentes con longitud máxima de {len(valid_dataloader)}")
   
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
                                 dropout=cfg.zmodel.model.decoder.dropout)
    decoder = decoder.to(device)

    model = module.EncoderDecoderWrapper(encoder=encoder,
                                    decoder_cell=decoder,
                                    output_size=len(Fout),
                                    output_sequence_len=FUTURO,
                                    device=device)
    model = model.to(device)

    # Funciones loss
    loss_fn = lf.LossFunction(**kwargs_loss)
    encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.zmodel.model.encoder.lr, weight_decay=cfg.zmodel.model.decoder.lr / 10)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=cfg.zmodel.model.decoder.lr, weight_decay=cfg.zmodel.model.decoder.lr / 10)
    if cfg.zmodel.model.scheduler:
        encoder_scheduler = optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=cfg.zmodel.model.encoder.lr, steps_per_epoch=len(train_dataloader), epochs=EPOCHS)
        decoder_scheduler = optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=cfg.zmodel.model.decoder.lr, steps_per_epoch=len(train_dataloader), epochs=EPOCHS)
        scheduler = [encoder_scheduler, decoder_scheduler]
    else:
        scheduler = None
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.zmodel.model.decoder.lr, weight_decay=cfg.zmodel.model.decoder.lr / 10)
    scheduler = optim.lr_scheduler.OneCycleLR(model_optimizer, max_lr=cfg.zmodel.model.decoder.lr, steps_per_epoch=len(train_dataloader), epochs=6)

    
    trainer = tr.TorchTrainer(name=name,
                              model=model,
                              optimizer=[encoder_optimizer, decoder_optimizer], 
                              loss_fn = loss_fn,   
                              scheduler=None,
                              device=device,
                              scheduler_batch_step=True if cfg.zmodel.model.scheduler else False,
                              checkpoint_folder= chkpts_path,
                              runs_folder= runs_path,
                              save_model=cfg.zmodel.model.save_model,
                              save_model_path=cfg.paths.zmodel.model,
                              early_stop=cfg.zmodel.model.early_stop,
                              alpha = cfg.zmodel.model.decoder.alpha
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
    
    try:
        with open(file, 'r') as handler:
            cfg = yaml.safe_load(handler)
            name = cfg["experiment"]
            cfg = AttrDict(parser(name, None)(cfg))
            print(f"Usando {file} como archivo de configuracion")
        with open(Path(cfg.paths.pmodel.dataset_metadata), 'r') as handler:
            metadata = yaml.safe_load(handler)
            print("Leidos metadatos del dataset")
        with open(Path(cfg.paths.pmodel.dataset), 'rb') as handler:
            datasets = pickle.load(handler)
            print(f"Usando {cfg.paths.pmodel.dataset} como archivo de datos procesados de estaciones")
    except Exception as e:
        print(f"El archivo de configuracion del experimento no existe o no existe el archivo {cfg.paths.pmodel.dataset} \
            con el dataset para el modelo zonal o {cfg.paths.pmodel.dataset_metadata} de metadatos del dataset del \
            modelo zonal. Mas info en: {e}")
        exit()
    device = 'cuda' if cfg.pmodel.model.use_cuda else 'cpu'
    Fout = len(cfg.prediccion)

    ## Generar Dataset de train
    try:
        train_dataset = pickle.load(open(Path(cfg.paths.pmodel.pmodel_train), "rb"))
        print(f"Cargango datos de train desde el modelo zmodel desde {cfg.paths.pmodel.pmodel_train}")
    except:
        print(f"Generando datos de train...")
        Path(cfg.paths.pmodel.pmodel_train).parent.mkdir(parents=True, exist_ok=True)
        kwargs_dataloader = generar_kwargs()._dataloader(modelo='pmodel', fase='train', cfg=cfg, datasets=datasets, metadata=metadata)
        dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
        y_real = np.empty((len(dataloader), cfg.futuro, Fout))
        pred_nwp = np.empty_like(y_real)
        for i, (_, _, _, Y, P) in enumerate(tqdm(dataloader)):
            # hay que quitarles la componente 0 y pasarlos a numpy porque son tensores
            y_real[i, ...] = Y[:, 1:, :].numpy()  # y_real = (len(dataset), Ly, Fout)
            pred_nwp[i, ...] = P[:, 1:, :].numpy()  # pred_nwp = (len(dataset), Ly, Fout)
        train_dataset = [pred_nwp, y_real]
        with open(Path(cfg.paths.pmodel.pmodel_train), 'wb') as handler:
            pickle.dump(train_dataset, handler)

    ## Generar Dataset de validacion
    try:
        valid_dataset = pickle.load(open(Path(cfg.paths.pmodel.pmodel_valid), "rb"))
        print(f"Cargango datos de validacion desde el modelo zmodel desde {cfg.paths.pmodel.pmodel_valid}")
    except:
        print(f"Generando datos de validacion...")
        Path(cfg.paths.pmodel.pmodel_valid).parent.mkdir(parents=True, exist_ok=True)
        kwargs_dataloader = generar_kwargs()._dataloader(modelo='pmodel', fase='validation', cfg=cfg, datasets=datasets, metadata=metadata)
        dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
        y_real = np.empty((len(dataloader), cfg.futuro, Fout))
        pred_nwp = np.empty_like(y_real)
        for i, (_, _, _, Y, P) in enumerate(tqdm(dataloader)):
            # hay que quitarles la componente 0 y pasarlos a numpy porque son tensores
            y_real[i, ...] = Y[:, 1:, :].numpy()  # y_real = (len(dataset), Ly, Fout)
            pred_nwp[i, ...] = P[:, 1:, :].numpy()  # pred_nwp = (len(dataset), Ly, Fout)
            
        valid_dataset = [pred_nwp, y_real]
        with open(Path(cfg.paths.pmodel.pmodel_valid), 'wb') as handler:
            pickle.dump(valid_dataset, handler)

    cfg_previo = copy.deepcopy(dict(cfg))
    if not temp and not hr and not rain:
        print("No se ha definido modelo para entrenar")
        exit()
    if temp:
        ## Parte especifica temperatura
        print("Entrenado modelo de temperatura...")
        cfg = AttrDict(parser(None, None, 'temperatura')(copy.deepcopy(cfg_previo)))
        EPOCHS = cfg.pmodel.model.temperatura.epochs
        kwargs_loss = cfg.pmodel.model.temperatura.loss_function
        TRAIN = cfg.zmodel.dataloaders.train.enable
        VALIDATION = cfg.zmodel.dataloaders.validation.enable
        print("")
        print(f"\tTrain: {SI if TRAIN else NO}, Validation: {SI if VALIDATION else NO}\n")
        print(f"Usando {cfg.paths.pmodel.runs} como ruta para runs")
        print(f"Usando {cfg.paths.pmodel.checkpoints} como ruta para guardar checkpoints")
        
        shuffle = False if 'shuffle' not in cfg.pmodel.dataloaders.train.keys() else cfg.pmodel.dataloaders.train.shuffle
        if TRAIN:
            train_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=train_dataset, componentes=slice(0, 1)),
                                        sampler=sa_pmodel.PModelSampler(datasets=train_dataset, batch_size=1, shuffle=shuffle),    
                                        batch_size=None,
                                        num_workers=2)
        shuffle = False if 'shuffle' not in cfg.pmodel.dataloaders.validation.keys() else cfg.pmodel.dataloaders.validation.shuffle
        if VALIDATION:
            valid_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=valid_dataset, componentes=slice(0, 1)),    
                                        sampler=sa_pmodel.PModelSampler(datasets=valid_dataset, batch_size=1, shuffle=shuffle),
                                        batch_size=None,
                                        num_workers=2)
            
        print(f"\tDataset de train: {len(train_dataloader)}")
        print(f"\tDataset de validacion: {len(valid_dataloader)}")

        model = md_pmodel.RedGeneral(Fin= 1, Fout=1, n_layers=2)
        model = model.to(device)
        ## Funciones loss
        kwargs_loss = cfg.pmodel.model.temperatura.loss_function
        loss_fn = lf.LossFunction(**kwargs_loss)
        model_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.pmodel.model.temperatura.lr, weight_decay=cfg.pmodel.model.temperatura.lr / 10)
        
        trainer = tr_pmodel.TorchTrainer(model=model, optimizer=model_optimizer, loss_fn = loss_fn, device=device,
                                checkpoint_folder= Path(cfg.paths.pmodel.checkpoints),
                                runs_folder= Path(cfg.paths.pmodel.runs),
                                save_model= cfg.pmodel.model.temperatura.save_model,
                                save_model_path=Path(cfg.paths.pmodel.model),
                                early_stop=cfg.pmodel.model.temperatura.early_stop)

        if TRAIN and VALIDATION:
            trainer.train(EPOCHS, train_dataloader, valid_dataloader, resume_only_model=True, resume=True, plot=cfg.pmodel.model.temperatura.plot_intermediate_results)
        else:
            trainer.train(EPOCHS, train_dataloader, resume_only_model=True, resume=True, plot=cfg.pmodel.model.temperatura.plot_intermediate_results)
        
        with open(Path(cfg.paths.pmodel.checkpoints) / "valid_losses.pickle", 'rb') as handler:
            valid_losses = pickle.load(handler)
        if valid_losses != {}:
            best_epoch = sorted(valid_losses.items(), key=lambda x:x[1])[0][0]
            print(f"Mejor loss de validacion: {best_epoch}")
    if hr:
        print("Entrando modelo de hr...")
        cfg = AttrDict(parser(None, None, 'hr')(copy.deepcopy(cfg_previo)))
        EPOCHS = cfg.pmodel.model.hr.epochs
        kwargs_loss = cfg.pmodel.model.hr.loss_function
        TRAIN = cfg.zmodel.dataloaders.train.enable
        VALIDATION = cfg.zmodel.dataloaders.validation.enable
        print(f"Train: {SI if TRAIN else NO}, Validation: {SI if VALIDATION else NO}\n")
        print(f"Usando {cfg.paths.pmodel.runs} como ruta para runs")
        print(f"Usando {cfg.paths.pmodel.checkpoints} como ruta para guardar checkpoints")
         
        shuffle = False if 'shuffle' not in cfg.pmodel.dataloaders.train.keys() else cfg.pmodel.dataloaders.train.shuffle
        if TRAIN:
            train_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=train_dataset, componentes=slice(1, 2)),
                                        sampler=sa_pmodel.PModelSampler(datasets=train_dataset, batch_size=1, shuffle=shuffle),    
                                        batch_size=None,
                                        num_workers=2)
        shuffle = False if 'shuffle' not in cfg.pmodel.dataloaders.validation.keys() else cfg.pmodel.dataloaders.validation.shuffle
        if VALIDATION:
            valid_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=valid_dataset, componentes=slice(1, 2)),    
                                        sampler=sa_pmodel.PModelSampler(datasets=valid_dataset, batch_size=1, shuffle=shuffle),
                                        batch_size=None,
                                        num_workers=2)
            
        print(f"\tDataset de train: {len(train_dataloader)}")
        print(f"\tDataset de validacion: {len(valid_dataloader)}")

        model = md_pmodel.RedGeneral(Fin= 1, Fout=1, n_layers=2)
        model = model.to(device)
        ## Funciones loss
        kwargs_loss = cfg.pmodel.model.hr.loss_function
        loss_fn = lf.LossFunction(**kwargs_loss)
        model_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.pmodel.model.hr.lr, weight_decay=cfg.pmodel.model.hr.lr / 10)
        
        trainer = tr_pmodel.TorchTrainer(model=model, optimizer=model_optimizer, loss_fn = loss_fn, device=device,
                                checkpoint_folder= Path(cfg.paths.pmodel.checkpoints),
                                runs_folder= Path(cfg.paths.pmodel.runs),
                                save_model= cfg.pmodel.model.temperatura.save_model,
                                save_model_path=Path(cfg.paths.pmodel.model),
                                early_stop=cfg.pmodel.model.temperatura.early_stop)

        if TRAIN and VALIDATION:
            trainer.train(EPOCHS, train_dataloader, valid_dataloader, resume_only_model=True, resume=True, plot=cfg.pmodel.model.hr.plot_intermediate_results)
        else:
            trainer.train(EPOCHS, train_dataloader, resume_only_model=True, resume=True, plot=cfg.pmodel.model.hr.plot_intermediate_results)
        
        with open(Path(cfg.paths.pmodel.checkpoints) / "valid_losses.pickle", 'rb') as handler:
            valid_losses = pickle.load(handler)
        if valid_losses != {}:
            best_epoch = sorted(valid_losses.items(), key=lambda x:x[1])[0][0]
            print(f"\tMejor loss de validacion: {best_epoch}")
    if rain:
        print("Entrando modelo de precipitacion...")
        cfg = AttrDict(parser(None, None, 'precipitacion')(copy.deepcopy(cfg_previo)))
        EPOCHS = cfg.pmodel.model.hr.epochs
        kwargs_loss = cfg.pmodel.model.hr.loss_function
        TRAIN = cfg.zmodel.dataloaders.train.enable
        VALIDATION = cfg.zmodel.dataloaders.validation.enable
        print(f"Train: {SI if TRAIN else NO}, Validation: {SI if VALIDATION else NO}\n")
        print(f"Usando {cfg.paths.pmodel.runs} como ruta para runs")
        print(f"Usando {cfg.paths.pmodel.checkpoints} como ruta para guardar checkpoints")
        shuffle = False if 'shuffle' not in cfg.pmodel.dataloaders.train.keys() else cfg.pmodel.dataloaders.train.shuffle
        if TRAIN:
            train_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=train_dataset, componentes=slice(2, 2 + len(metadata["bins"]))),
                                        sampler=sa_pmodel.PModelSampler(datasets=train_dataset, batch_size=1, shuffle=shuffle),    
                                        batch_size=None,
                                        num_workers=2)
        shuffle = False if 'shuffle' not in cfg.pmodel.dataloaders.validation.keys() else cfg.pmodel.dataloaders.validation.shuffle
        if VALIDATION:
            valid_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=valid_dataset, componentes=slice(2, 2 + len(metadata["bins"]))),    
                                        sampler=sa_pmodel.PModelSampler(datasets=valid_dataset, batch_size=1, shuffle=shuffle),
                                        batch_size=None,
                                        num_workers=2)
            
        print(f"\tDataset de train: {len(train_dataloader)}")
        print(f"\tDataset de validacion: {len(valid_dataloader)}")

        model = md_pmodel.RedGeneral(Fin= 8, Fout=8, n_layers=2)
        model = model.to(device)
        ## Funciones loss
        kwargs_loss = cfg.pmodel.model.hr.loss_function
        weights = torch.tensor([1/150541, 1/1621, 1/512, 1/249, 1/176, 1/121, 1/46, 1/1]).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        model_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.pmodel.model.precipitacion.lr, weight_decay=cfg.pmodel.model.precipitacion.lr / 10)
        
        trainer = tr_pmodel.TorchTrainer(model=model, optimizer=model_optimizer, loss_fn = loss_fn, device=device,
                                checkpoint_folder= Path(cfg.paths.pmodel.checkpoints),
                                runs_folder= Path(cfg.paths.pmodel.runs),
                                save_model= cfg.pmodel.model.temperatura.save_model,
                                save_model_path=Path(cfg.paths.pmodel.model),
                                early_stop=cfg.pmodel.model.temperatura.early_stop)

        if TRAIN and VALIDATION:
            trainer.train(EPOCHS, train_dataloader, valid_dataloader, resume_only_model=True, resume=True, rain=True, plot=cfg.pmodel.model.precipitacion.plot_intermediate_results)
        else:
            trainer.train(EPOCHS, train_dataloader, resume_only_model=True, resume=True, rain=True, plot=cfg.pmodel.model.precipitacion.plot_intermediate_results)
        
        with open(Path(cfg.paths.pmodel.checkpoints) / "valid_losses.pickle", 'rb') as handler:
            valid_losses = pickle.load(handler)
        if valid_losses != {}:
            best_epoch = sorted(valid_losses.items(), key=lambda x:x[1])[0][0]
            print(f"\tMejor loss de validacion: {best_epoch}")

if __name__ == "__main__":
    main()
