import click
import yaml
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from attrdict import AttrDict
from tqdm import tqdm
import importlib

from colorama import Fore, Back, Style
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from common.utils.datasets import experimento1dataset as ds
from common.utils.datasets import experimento1sampler as sa
from common.utils.trainers import experimento1trainer as tr
from common.utils.loss_functions import lossfunction as lf


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def predict(**kwargs):

    datasets = kwargs.get('datasets', None)
    fecha_inicio_test = kwargs.get('fecha_inicio_test', None)
    fecha_fin_test = kwargs.get('fecha_fin_test', None)
    pasado = kwargs.get('pasado', None)
    futuro = kwargs.get('futuro', None)
    etiquetaX = kwargs.get('etiquetaX', None)
    etiquetaF = kwargs.get('etiquetaF', None)
    etiquetaT = kwargs.get('etiquetaT', None)
    etiquetaP = kwargs.get('etiquetaP', None)
    name = kwargs.get('name', None)
    model_name = kwargs.get('model_name', None)
    rnn_num_layers = kwargs.get('rnn_num_layers', None)
    encoder_hidden_size = kwargs.get('encoder_hidden_size', None)
    encoder_bidirectional = kwargs.get('encoder_bidirectional', None)
    device = kwargs.get('device', None)
    encoder_rnn_dropout = kwargs.get('encoder_rnn_dropout', None)
    decoder_hidden_size = kwargs.get('decoder_hidden_size', None)
    decoder_dropout = kwargs.get('decoder_dropout', None)
    model_scheduler = kwargs.get('model_scheduler', None)
    path_checkpoints = kwargs.get('path_checkpoints', None)
    use_checkpoint = kwargs.get('use_checkpoint', None)
    epochs = kwargs.get('epochs', None)
    inicio = kwargs.get('fecha_inicio', None)
    path_model = kwargs.get('path_model', None)
    indice_min = kwargs.get('indice_min', None)
    indice_max = kwargs.get('indice_max', None)
    
    # inicio = min([_ for _ in [datetime.strptime(fecha_inicio_train, "%Y-%m-%d %H:%M:%S"),
    #                           datetime.strptime(fecha_inicio_validation, "%Y-%m-%d %H:%M:%S")]
    #               if _ is not None])
    
    x = (fecha_inicio_test - inicio).days * 24 + (fecha_inicio_test - inicio).seconds / 3600
    y = (fecha_fin_test - inicio).days * 24 + (fecha_fin_test - inicio).seconds / 3600
    print(f"Generando dataset de test desde {x} a {y}")
    
    # dfs_test = [_.loc[(_.index >= x ) & (_.index <= y), :] for _ in datasets] con esta instruccion
    # estamos limitando el datset. La primera prediccion sera desde la fecha indicada + PASADO (de esto
    # se encargan los samplers) por lo que tendremos que añadir PASADO para que se empiece a predecir en la
    # fecha que hemos indicado. Lo mismo pasa con FUTURO
    
    if x - pasado < indice_min:
        print(Fore.YELLOW + f"No hay datos pasados para predicir desde la fecha indicada" + Style.RESET_ALL)
        exit()
    else: 
        x = x - pasado
    if y + futuro > indice_max:
        print(Fore.RED + f"No hay datos futuros para predicir hasta la fecha indicada" + Style.RESET_ALL)
        exit()
    else:
        y = y + futuro
 
    dfs_test = [_.loc[(_.index >= x ) & (_.index <= y), :] for _ in datasets]
    test_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(datasets=dfs_test,
                                                            pasado=pasado,
                                                            futuro=futuro,
                                                            etiquetaX=etiquetaX,
                                                            etiquetaF=etiquetaF,
                                                            etiquetaT=etiquetaT,
                                                            etiquetaP=etiquetaP),
                                  sampler=sa.Seq2SeqSampler(datasets=dfs_test,
                                                            pasado=pasado,
                                                            futuro=futuro,
                                                            shuffle=False),

                                  batch_size=None,
                                  num_workers=8)
    # print(f"Dataset de test: {len(dfs_test)} componentes con longitud máxima de {len(test_dataloader)}")

    module = importlib.import_module(f"models.seq2seq.{model_name}")
    encoder = module.RNNEncoder(rnn_num_layers=rnn_num_layers,
                            input_feature_len=len(etiquetaF) + 1,
                            sequence_len=pasado + 1,
                            hidden_size=encoder_hidden_size,
                            bidirectional=encoder_bidirectional,
                            device=device,
                            rnn_dropout=encoder_rnn_dropout)
    encoder = encoder.to(device)

    decoder = module.DecoderCell(input_feature_len=len(etiquetaT) + 1,
                            hidden_size=decoder_hidden_size,
                            dropout=decoder_dropout)
    decoder = decoder.to(device)
    encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=1e-3)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=1e-3)
    if model_scheduler:
        encoder_scheduler = optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=1e-3, steps_per_epoch=len(test_dataloader), epochs=epochs)
        decoder_scheduler = optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=1e-3, steps_per_epoch=len(test_dataloader), epochs=epochs)
        scheduler = [encoder_scheduler, decoder_scheduler]
    else:
        scheduler = None
    model = torch.load(Path(path_model) , map_location='cpu')
    model.to(device)

    trainer = tr.TorchTrainer(name=name,
                              model=model,
                              optimizer=[encoder_optimizer, decoder_optimizer], 
                              loss_fn = None,   
                              scheduler=scheduler,
                              device=device,
                              scheduler_batch_step=True if model_scheduler else False,
                              pass_y=True,
                              checkpoint_folder= Path(path_checkpoints),
                              )
    # cargar checkpoint
    if use_checkpoint == 'best':
        trainer._load_best_checkpoint()
    else:
        trainer._load_checkpoint(epoch=use_checkpoint, only_model=True)
    y_pred = trainer.predict(test_dataloader)  # y_pred (len(test), N, L, F(d)out) (4000,1,72,1)
    return y_pred, test_dataloader