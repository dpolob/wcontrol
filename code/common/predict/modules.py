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
from torch.utils.data import DataLoader, dataloader

from common.utils.datasets import dataset_seq2seq as ds
from common.utils.datasets import sampler_seq2seq as sa
from common.utils.trainers import trainerSeq2Seq as tr
from common.utils.loss_functions import lossfunction as lf


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def generar_test_dataset(**kwargs) -> DataLoader:
    """Funcion para generar el dataset

    Returns:
        Dataloader 
    """
   
    datasets = kwargs.get('datasets', None)
    fecha_inicio_test = kwargs.get('fecha_inicio_test', None)
    fecha_fin_test = kwargs.get('fecha_fin_test', None)
    pasado = kwargs.get('pasado', None)
    futuro = kwargs.get('futuro', None)
    Fout = kwargs.get('etiquetaX', None)
    Ff = kwargs.get('etiquetaF', None)
    Ft = kwargs.get('etiquetaT', None)
    Fnwp = kwargs.get('etiquetaP', None)
    inicio = kwargs.get('fecha_inicio', None)
    indice_min = kwargs.get('indice_min', None)
    indice_max = kwargs.get('indice_max', None)
 
    x = (fecha_inicio_test - inicio).days * 24 + (fecha_inicio_test - inicio).seconds / 3600
    y = (fecha_fin_test - inicio).days * 24 + (fecha_fin_test - inicio).seconds / 3600
    print(f"Generando dataset de test desde {x} a {y}")
    
    # dfs_test = [_.loc[(_.index >= x ) & (_.index <= y), :] for _ in datasets] con esta instruccion
    # estamos limitando el dataset. La primera prediccion sera desde la fecha indicada + PASADO (de esto
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
                                                            etiquetaX=Fout,
                                                            etiquetaF=Ff,
                                                            etiquetaT=Ft,
                                                            etiquetaP=Fnwp),
                                  sampler=sa.Seq2SeqSampler(datasets=dfs_test,
                                                            pasado=pasado,
                                                            futuro=futuro,
                                                            shuffle=False),

                                  batch_size=None,
                                  num_workers=8)
    return test_dataloader


def predict(test_dataloader: DataLoader=None, **kwargs) -> np.ndarray:
    """Carga el modelo y realiza la prediccion con el dataset especificado

    Args:
        test_dataloader (DataLoader, optional): Datos sobre los que realizar la predicción. Defaults to None.

    Returns:
        [nd.array]: Prediccion con shape(len(test), N, Ly, Fout)
    """

    name = kwargs.get('name', None)
    device = kwargs.get('device', None)
    path_checkpoints = kwargs.get('path_checkpoints', None)
    use_checkpoint = kwargs.get('use_checkpoint', None)
    path_model = kwargs.get('path_model', None)
   
    model = torch.load(Path(path_model) , map_location='cpu')
    model.to(device)

    trainer = tr.TorchTrainer(name=name,
                              model=model,
                              device=device,
                              checkpoint_folder= Path(path_checkpoints),
                              )
   
    if use_checkpoint == 'best':
        trainer._load_best_checkpoint()
    else:
        trainer._load_checkpoint(epoch=use_checkpoint, only_model=True)
    y_pred = trainer.predict(test_dataloader)  # se devuelve una lista de numpy (len(test), N, Ly, Fout), dataloader
    return np.array(y_pred)