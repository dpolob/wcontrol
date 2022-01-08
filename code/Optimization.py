import click
import yaml
import pickle
import numpy as np

from datetime import datetime
from attrdict import AttrDict
from colorama import Fore, Back, Style
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from common.utils.parser import parser
from common.utils.datasets import dataset_seq2seq as ds
from common.utils.datasets import dataset_pmodel as ds_pmodel
from common.utils.datasets import sampler_seq2seq as sa
from common.utils.datasets import sampler_pmodel as sa_pmodel
from common.utils.loss_functions import lossfunction as lf
from common.optim.zmodel import optim_seq2seq_2layers_2heads as md
from common.optim.pmodel import optim_pmodel as md_pmodel
from common.optim.zmodel import optim_trainerSeq2Seq as tr
from common.optim.pmodel import optim_trainerpmodel as tr_pmodel

import optuna
from optuna.trial import TrialState

archivo, mtemp, mhr, mrain = None, None, None, None
torch.manual_seed(420)
np.random.seed(420)

def objective(trial):
    global archivo
    
    # Definicion de los hiperparametos a optimizar
    encoder_rnn_num_layers = trial.suggest_int('encoder_rnn_num_layers', 1, 2)
    hidden_size = trial.suggest_discrete_uniform('hidden_size', 50, 200, 50)
    encoder_bidirectional = trial.suggest_categorical("encoder_bidirectional", [True, False])
    encoder_dropout = trial.suggest_float('encoder_dropout', 0.0, 0.3)
    
    decoder_temphr_n_layers = trial.suggest_int('decoder_temphr_n_layers', 1, 3)
    decoder_class_n_layers = trial.suggest_int('decoder_temphr_n_layers', 1, 3)
    decoder_dropout = trial.suggest_float('decoder_dropout', 0.0, 0.3)
    
    #dataset_pasado = trial.suggest_int(dataset_pasado', 10, 20)
    dataset_pasado = trial.suggest_int('dataset_pasado', 72, 900)
    trainer_alpha = trial.suggest_float('trainer_alpha', 0.01, 0.99)
    encoder_optimizer_name = trial.suggest_categorical("encoder_optimizer_name", ["Adam", "AdamW", "RMSprop", "Adamax"])
    decoder_optimizer_name = trial.suggest_categorical("decoder_optimizer_name", ["Adam", "AdamW", "RMSprop", "Adamax"])
   
    encoder_lr = trial.suggest_float("encoder_lr", 0.001, 0.1, log=True)
    decoder_lr = trial.suggest_float("decoder_lr", 0.001, 0.1, log=True)
    # Lectura de variables desde archivo
    try:
        with open(Path(archivo), 'r') as handler:
            cfg = yaml.safe_load(handler)
            name = cfg["experiment"]
            cfg = AttrDict(parser(name, None)(cfg))
        with open(Path(cfg.paths.zmodel.dataset_metadata), 'r') as handler:
            metadata = yaml.safe_load(handler)
        with open(Path(cfg.paths.zmodel.dataset), 'rb') as handler:
            datasets = pickle.load(handler)
    except:
        print(f"{archivo} no existe o Datasets no existen")
        exit()
    
    
    device = 'cuda' if cfg.zmodel.model.use_cuda else 'cpu'
    FUTURO = cfg.futuro
    EPOCHS = 6  # Defino 5 como un valor máximo
    kwargs_loss = cfg.zmodel.model.loss_function
    Fout = list(cfg.prediccion)  # etiquetas a predecir len = Fout
    Ff = list(cfg.zmodel.model.encoder.features)  # etiquetas para encoder len = Ff
    Ft = list(cfg.zmodel.model.decoder.features)  # etiquetas para decoder len = Ft
    Fnwp = list(cfg.zmodel.model.decoder.nwp)  # etiquetas de los modelos nwp len = Fnwp
    fecha_inicio = datetime.strptime(metadata['fecha_min'], "%Y-%m-%d %H:%M:%S")
    shuffle = True if 'shuffle' not in cfg.zmodel.dataloaders.train.keys() else cfg.zmodel.dataloaders.train.shuffle
    
    TRAIN = cfg.zmodel.dataloaders.train.enable
    if TRAIN:
        FECHA_INICIO_TRAIN = datetime.strptime(cfg.zmodel.dataloaders.train.fecha_inicio, "%Y-%m-%d %H:%M:%S")
        FECHA_FIN_TRAIN = datetime.strptime(cfg.zmodel.dataloaders.train.fecha_fin, "%Y-%m-%d %H:%M:%S")
        x = (FECHA_INICIO_TRAIN - fecha_inicio).days * 24 + (FECHA_INICIO_TRAIN - fecha_inicio).seconds / 3600
        y = (FECHA_FIN_TRAIN - fecha_inicio).days * 24 + (FECHA_FIN_TRAIN - fecha_inicio).seconds / 3600
        dfs_train = [df.loc[(df.index >= x ) & (df.index <= y), :] for df in datasets]
    
    VALIDATION = cfg.zmodel.dataloaders.validation.enable
    if VALIDATION:
        FECHA_INICIO_VALID = datetime.strptime(cfg.zmodel.dataloaders.validation.fecha_inicio, "%Y-%m-%d %H:%M:%S")
        FECHA_FIN_VALID = datetime.strptime(cfg.zmodel.dataloaders.validation.fecha_fin, "%Y-%m-%d %H:%M:%S")
        x = (FECHA_INICIO_VALID - fecha_inicio).days * 24 + (FECHA_INICIO_VALID - fecha_inicio).seconds / 3600
        y = (FECHA_FIN_VALID - fecha_inicio).days * 24 + (FECHA_FIN_VALID - fecha_inicio).seconds / 3600
        dfs_valid = [df.loc[(df.index >= x ) & (df.index <= y), :] for df in datasets]
   
    
    train_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(datasets=dfs_train,
                                                            pasado=dataset_pasado,
                                                            futuro=FUTURO,
                                                            etiquetaX=Fout,
                                                            etiquetaF=Ff,
                                                            etiquetaT=Ft,
                                                            etiquetaP=Fnwp),
                                  sampler=sa.Seq2SeqSampler(datasets=dfs_train,
                                                            pasado=dataset_pasado,
                                                            futuro=FUTURO,
                                                            shuffle=shuffle),
                                  batch_size=None,
                                  num_workers=8)
    shuffle = True if 'shuffle' not in cfg.zmodel.dataloaders.validation.keys() else cfg.zmodel.dataloaders.validation.shuffle
    valid_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(datasets=dfs_valid,
                                                                pasado=dataset_pasado,
                                                                futuro=FUTURO,
                                                                etiquetaX=Fout,
                                                                etiquetaF=Ff,
                                                                etiquetaT=Ft,
                                                                etiquetaP=Fnwp),
                                   sampler=sa.Seq2SeqSampler(datasets=dfs_valid,
                                                                pasado=dataset_pasado,
                                                                futuro=FUTURO,
                                                                shuffle=shuffle),
                                   batch_size=None,
                                   num_workers=8)
   

   
    encoder = md.RNNEncoder(rnn_num_layers=encoder_rnn_num_layers,
                            input_feature_len=len(Ff),
                            sequence_len=dataset_pasado + 1, # +1 porque tomamos tiempo0
                            hidden_size=hidden_size,
                            bidirectional=encoder_bidirectional,
                            device=device,
                            rnn_dropout=encoder_dropout)
    encoder = encoder.to(device)

    decoder = md.DecoderCell(input_feature_len=len(Ft) + len(Fnwp),
                             hidden_size=hidden_size,
                             output_size=len(Fout),
                             bins_len= 8,
                             dropout=decoder_dropout,
                             decoder_temphr_n_layers=decoder_temphr_n_layers,
                             decoder_class_n_layers=decoder_class_n_layers,
                             trial=trial)
    decoder = decoder.to(device)

    model = md.EncoderDecoderWrapper(encoder=encoder,
                                     decoder_cell=decoder,
                                     output_size=len(Fout),
                                     output_sequence_len=FUTURO,
                                     device=device)
    model = model.to(device)

    # Funciones loss
    loss_fn = lf.LossFunction(**kwargs_loss)
    
    encoder_optimizer = getattr(optim, encoder_optimizer_name)(encoder.parameters(), lr=encoder_lr, weight_decay=encoder_lr / 10)
    decoder_optimizer = getattr(optim, decoder_optimizer_name)(decoder.parameters(), lr=decoder_lr, weight_decay=decoder_lr / 10)

    trainer = tr.TorchTrainer(name=name,
                              model=model,
                              optimizer=[encoder_optimizer, decoder_optimizer], 
                              loss_fn=loss_fn,   
                              device=device,
                              alpha = trainer_alpha
                              )

    start_epoch = 0
    for i_epoch in range(start_epoch, start_epoch + EPOCHS):
        model.train()
        #train_bar = tqdm(train_dataloader)
        for (Xf, X, Yt, Y, P) in train_dataloader:
            loss = trainer._loss_batch(Xf, X, Yt, Y, P, optimize=True)
        print(f"Loss {i_epoch}: {loss}")    
        
        model.eval()
        loss_values = []
        #valid_bar = tqdm(valid_dataloader)
        for Xf, X, Yt, Y, P in valid_dataloader:
            loss_value = trainer._loss_batch(Xf, X, Yt, Y, P, optimize=False)
            loss_values.append(loss_value)
                # valid_bar.set_description(f"V_Loss: {loss_value}")        
        print(f"V_Loss {i_epoch}: {loss_value}")
        loss_value = np.mean(loss_values)
                    
        trial.report(loss_value, i_epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return loss_value

def objectivepmodel(trial):
    global archivo; global mtemp; global mhr; global mrain 
    
    n_layers = trial.suggest_int('n_layers', 1, 5)
    hidden_size = trial.suggest_discrete_uniform('hidden_size', 50, 200, 25)
    gru_n_layers = trial.suggest_int('gru_n_layers', 1, 3)
    model_optimizer_name = trial.suggest_categorical("loss_function", ["Adam", "AdamW", "RMSprop", "Adamax"])
    lr = trial.suggest_float("lr", 0.001, 0.1, log=True)
  
  
    # Lectura de variables desde archivo
    try:
        with open(archivo, 'r') as handler:
            cfg = yaml.safe_load(handler)
            name = cfg["experiment"]
            cfg = AttrDict(parser(name, None)(cfg))
        with open(Path(cfg.paths.pmodel.dataset_metadata), 'r') as handler:
            metadata = yaml.safe_load(handler)
        with open(Path(cfg.paths.pmodel.dataset), 'rb') as handler:
            datasets = pickle.load(handler)
    except:
        print(f"{archivo} no existe o no existe el archivo de metadatos o no existe el archivo de datos de estaciones")
        exit()
    device = 'cuda' if cfg.pmodel.model.use_cuda else 'cpu'
    
    ## Cargar Datasets
    try:
        train_dataset = pickle.load(open(Path(cfg.paths.pmodel.pmodel_train), "rb"))
        valid_dataset = pickle.load(open(Path(cfg.paths.pmodel.pmodel_valid), "rb"))
    except:
        print("No se han cargado los dataset... por favor generarlos antes")
        exit()
    
    def entrenador_optimo(Fin: int, Fout: int, componente: slice, kwargs_loss: dict, cfg=cfg):
        ## Parte especifica temperatura
        cfg = AttrDict(parser(None, None, 'temperatura')(dict(cfg)))
        EPOCHS = 25
        TRAIN = cfg.zmodel.dataloaders.train.enable
        VALIDATION = cfg.zmodel.dataloaders.validation.enable
        shuffle = True if 'shuffle' not in cfg.pmodel.dataloaders.train.keys() else cfg.pmodel.dataloaders.train.shuffle
        if TRAIN:
            train_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=train_dataset, componentes=componente),
                                        sampler=sa_pmodel.PModelSampler(datasets=train_dataset, batch_size=1, shuffle=shuffle),    
                                        batch_size=None,
                                        num_workers=8)
        shuffle = True if 'shuffle' not in cfg.pmodel.dataloaders.validation.keys() else cfg.pmodel.dataloaders.validation.shuffle
       
        if VALIDATION:
            valid_dataloader = DataLoader(dataset=ds_pmodel.PModelDataset(datasets=valid_dataset, componentes=componente),    
                                        sampler=sa_pmodel.PModelSampler(datasets=valid_dataset, batch_size=1, shuffle=shuffle),
                                        batch_size=None,
                                        num_workers=8)
            
        model = md_pmodel.RedGeneral(Fin=Fin, Fout=Fout, n_layers=n_layers, hidden_size=hidden_size, gru_num_layers=gru_n_layers, trial=trial)
        model = model.to(device)

        ## Funciones loss
        loss_fn = lf.LossFunction(**kwargs_loss)
        model_optimizer = getattr(optim, model_optimizer_name)(model.parameters(), lr=lr, weight_decay=lr / 10)
        
        trainer = tr_pmodel.TorchTrainer(name=name, model=model, optimizer=model_optimizer, loss_fn=loss_fn, device=device)

        start_epoch = 0
        for i_epoch in range(start_epoch, start_epoch + EPOCHS):
            model.train()

            for (X, Y) in train_dataloader:
                loss = trainer._loss_batch(X, Y, optimize=True)
            print(f"Loss {i_epoch}: {loss}")    
        
            model.eval()
            loss_values = []
            for X, Y in valid_dataloader:
                loss_value = trainer._loss_batch(X, Y, optimize=False)
                loss_values.append(loss_value)
            loss_value = np.mean(loss_values)
            print(f"V_Loss {i_epoch}: {loss_value}")
                    
        trial.report(loss_value, i_epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return loss_value
    
    if mtemp:
        return entrenador_optimo(Fin=1, Fout=1, componente=slice(0, 1), kwargs_loss=cfg.pmodel.model.temperatura.loss_function)
    if mhr:
        return entrenador_optimo(Fin=1, Fout=1,componente=slice(1, 2), kwargs_loss=cfg.pmodel.model.hr.loss_function)
    if mrain:
        return entrenador_optimo(Fin=len(metadata["bins"]), Fout=len(metadata["bins"]), componente=slice(2, 2 + len(metadata["bins"]), kwargs_loss=cfg.pmodel.model.precipitacion.loss_function))
    
    
    
@click.group()
def main():
    pass

@main.command()
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
def zmodel(file):
    global archivo 
    archivo = file
    study_name = "zmodel"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    #storage_name = "mysql://root:root@127.0.0.1:3306/prueba"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
    study.optimize(objective, n_trials=50)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

@main.command()
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
@click.option('--temp', is_flag=True, default= False, help='Entrenar modelo de temperatura')
@click.option('--hr', is_flag=True, default= False, help='Entrenar modelo de HR')
@click.option('--rain', is_flag=True, default= False, help='Entrenar modelo de precipitacion')
def pmodel(file, temp, hr, rain):
    global archivo; global mtemp; global mhr; global mrain 
    archivo = file
    
    def instancia(nombre:str) -> None:
        study_name = f"pmodel_{nombre}"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        #storage_name = "mysql://root:root@127.0.0.1:3306/prueba"
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
        study.optimize(objectivepmodel, n_trials=100)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    
    if temp:
       mtemp = True; mhr = False; mrain = False 
       instancia("temp")
    if hr:
       mtemp = False; mhr = True; mrain = False 
       instancia("hr")
    if rain:
       mtemp = False; mhr = False; mrain = True 
       instancia("rain")
            
if __name__ == "__main__":
    main()