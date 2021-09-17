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


def predictor(**kwargs):

    datasets = kwargs.get('datasets', None)
    fecha_inicio_test = kwargs.get('fecha_inicio_test', None)
    fecha_fin_test = kwargs.get('fecha_fin_test', None)
    fecha_inicio_train = kwargs.get('fecha_inicio_train', None)
    fecha_inicio_validation =  kwargs.get('fecha_inicio_train', None)
    pasado = kwargs.get('pasado', None)
    futuro = kwargs.get('futuro', None)
    etiquetaX = kwargs.get('etiquetaX', None)
    etiquetaF = kwargs.get('etiquetaF', None)
    etiquetaT = kwargs.get('etiquetaT', None)
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
    
    inicio = min([_ for _ in [datetime.strptime(fecha_inicio_train, "%Y-%m-%d %H:%M:%S"),
                              datetime.strptime(fecha_inicio_validation, "%Y-%m-%d %H:%M:%S")]
                  if _ is not None])
    
    x = (fecha_inicio_test - inicio).days * 24 + (fecha_inicio_test - inicio).seconds / 3600
    y = (fecha_fin_test - inicio).days * 24 + (fecha_fin_test - inicio).seconds / 3600
    print(f"Generando dataset de test desde {x} a {y}")
       
    dfs_test = [_.loc[(_.index >= x ) & (_.index <= y), :] for _ in datasets]

    test_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(datasets=dfs_test,
                                                            pasado=pasado,
                                                            futuro=futuro,
                                                            etiquetaX=etiquetaX,
                                                            etiquetaF=etiquetaF,
                                                            etiquetaT=etiquetaT),
                                  sampler=sa.Seq2SeqSampler(datasets=dfs_test,
                                                            pasado=pasado,
                                                            futuro=futuro,
                                                            shuffle=False),

                                  batch_size=None,
                                  num_workers=8)
    print(f"Dataset de test: {len(dfs_test)} componentes con longitud máxima de {len(test_dataloader)}")

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
    model = torch.load(Path(path_checkpoints) / name / "model.pth" , map_location='cpu')
    model.to(device)
    trainer = tr.TorchTrainer(name=name,
                              model=model,
                              optimizer=[encoder_optimizer, decoder_optimizer], 
                              loss_fn = None,   
                              scheduler=scheduler,
                              device=device,
                              scheduler_batch_step=True if model_scheduler else False,
                              pass_y=True,
                              checkpoint_folder= Path(path_checkpoints) / name,
                              )
    # cargar checkpoint
    if use_checkpoint == 'best':
        trainer._load_best_checkpoint()
    else:
        trainer._load_checkpoint(epoch=use_checkpoint, only_model=True)
    y_pred = trainer.predict(test_dataloader)  # y_pred (len(test), N, L, F(d)out) (4000,1,72,1)
    return y_pred, test_dataloader
                                           
@click.command()
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
def predictZModel(file):

    try:
        with open(file, 'r') as handler:
            cfg = AttrDict(yaml.safe_load(handler))
        print(f"Usando {file} como archivo de configuracion")
    except:
        print(f"{file} no existe. Por favor defina un archivo con --file")
    
    try:
        with open(cfg.paths.datasets, 'rb') as handler:
            datasets = pickle.load(handler)
        print(f"Usando {cfg.paths.datasets} como archivo de datos procesados de estaciones")
    except:
        print(Fore.RED + "Por favor defina un archivo de datos procesados")
    
    name = cfg.experiment
    output = Path(cfg.paths.predictions) / name / "prediction.pickle"
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Usando {output} como ruta para guardar predicciones")

    PASADO = cfg.entrenamiento.pasado
    FUTURO = cfg.entrenamiento.futuro
    PREDICCION = cfg.entrenamiento.prediccion
    device = 'cuda' if cfg.prediccion.use_cuda else 'cpu'        
    TEST = cfg.prediccion.dataloaders.test.enable
    FECHA_INICIO_TEST = None
    FECHA_FIN_TEST = None
    FEATURES = list(cfg.entrenamiento.features)
    DEFINIDAS = list(cfg.entrenamiento.definidas)
    EPOCHS = cfg.entrenamiento.epochs
    
    if TEST:
        FECHA_INICIO_TEST = datetime.strptime(cfg.prediccion.dataloaders.test.fecha_inicio, "%Y-%m-%d %H:%M:%S")
        FECHA_FIN_TEST = datetime.strptime(cfg.prediccion.dataloaders.test.fecha_fin, "%Y-%m-%d %H:%M:%S")
    
        # inicio = min([_ for _ in [datetime.strptime(cfg.entrenamiento.dataloaders.train.fecha_inicio, "%Y-%m-%d %H:%M:%S"),
        #                           datetime.strptime(cfg.entrenamiento.dataloaders.validation.fecha_inicio, "%Y-%m-%d %H:%M:%S")]
        #               if _ is not None])
    
        # x = (FECHA_INICIO_TEST - inicio).days * 24 + (FECHA_INICIO_TEST - inicio).seconds / 3600
        # y = (FECHA_FIN_TEST - inicio).days * 24 + (FECHA_FIN_TEST - inicio).seconds / 3600
        # #x=2001; y=2500
        # print(f"Generando dataset de test desde {x} a {y}")
    
        # dfs_test = [df.loc[(df.index >= x ) & (df.index <= y), :] for df in datasets]

    
        # test_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(datasets=dfs_test,
        #                                                         pasado=PASADO,
        #                                                         futuro=FUTURO,
        #                                                         etiquetaX=PREDICCION,
        #                                                         etiquetaF=FEATURES,
        #                                                         etiquetaT=DEFINIDAS),
        #                               sampler=sa.Seq2SeqSampler(datasets=dfs_test,
        #                                                         pasado=PASADO,
        #                                                         futuro=FUTURO,
        #                                                         shuffle=False),
    
        #                               batch_size=None,
        #                               num_workers=8)
        # # for a,b,c,d in test_dataloader:
        # #     print(a.shape, b.shape, c.shape, d.shape)
        # # exit()
        # print(f"Dataset de test: {len(dfs_test)} componentes con longitud máxima de {len(test_dataloader)}")


    
    # module = importlib.import_module(f"models.seq2seq.{cfg.entrenamiento.model.name}")
    # encoder = module.RNNEncoder(rnn_num_layers=cfg.entrenamiento.encoder.rnn_num_layers,
    #                         input_feature_len=len(FEATURES) + 1,
    #                         sequence_len=PASADO + 1,
    #                         hidden_size=cfg.entrenamiento.encoder.hidden_size,
    #                         bidirectional=cfg.entrenamiento.encoder.bidirectional,
    #                         device=device,
    #                         rnn_dropout=cfg.entrenamiento.encoder.rnn_dropout)
    # encoder = encoder.to(device)

    # decoder = module.DecoderCell(input_feature_len=len(DEFINIDAS) + 1,
    #                         hidden_size=cfg.entrenamiento.decoder.hidden_size,
    #                         dropout=cfg.entrenamiento.decoder.dropout)
    # decoder = decoder.to(device)
    # encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=1e-3)
    # decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=1e-3)
    # if cfg.entrenamiento.scheduler:
    #     encoder_scheduler = optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=1e-3, steps_per_epoch=len(test_dataloader), epochs=EPOCHS)
    #     decoder_scheduler = optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=1e-3, steps_per_epoch=len(test_dataloader), epochs=EPOCHS)
    #     scheduler = [encoder_scheduler, decoder_scheduler]
    # else:
    #     scheduler = None
        
    # model = torch.load(Path(cfg.paths.checkpoints) / name / "model.pth" , map_location='cpu')
    # model.to(device)
    # trainer = tr.TorchTrainer(name=name,
    #                           model=model,
    #                           optimizer=[encoder_optimizer, decoder_optimizer], 
    #                           loss_fn = None,   
    #                           scheduler=scheduler,
    #                           device=device,
    #                           scheduler_batch_step=True if cfg.entrenamiento.scheduler else False,
    #                           pass_y=True,
    #                           checkpoint_folder= Path(cfg.paths.checkpoints) / name,
    #                           )
    # # cargar mejor checkpoint
    # if cfg.prediccion.use_checkpoint == 'best':
    #     trainer._load_best_checkpoint()
    # else:
    #     trainer._load_checkpoint(epoch=cfg.prediccion.use_checkpoint, only_model=True)
    # y_pred = trainer.predict(test_dataloader)  # y_pred (len(test), N, L, F(d)out) (4000,1,72,1)
 
    kwargs =   {'datasets': datasets,
     'fecha_inicio_test': FECHA_INICIO_TEST,
      'fecha_fin_test': FECHA_FIN_TEST,
      'fecha_inicio_train': cfg.entrenamiento.dataloaders.train.fecha_inicio,
      'fecha_inicio_validation':  cfg.entrenamiento.dataloaders.validation.fecha_inicio,
      'pasado': PASADO,
      'futuro': FUTURO,
      'etiquetaX': PREDICCION,
      'etiquetaF': FEATURES,
      'etiquetaT': DEFINIDAS,
      'name': name,
      'model_name': cfg.entrenamiento.model.name,
      'rnn_num_layers': cfg.entrenamiento.encoder.rnn_num_layers,
      'encoder_hidden_size': cfg.entrenamiento.encoder.hidden_size,
      'encoder_bidirectional' : cfg.entrenamiento.encoder.bidirectional,
      'device': device,
      'encoder_rnn_dropout': cfg.entrenamiento.encoder.rnn_dropout,
      'decoder_hidden_size': cfg.entrenamiento.decoder.hidden_size,
      'decoder_dropout': cfg.entrenamiento.decoder.dropout,
      'model_scheduler': cfg.entrenamiento.scheduler,
      'path_checkpoints': cfg.paths.checkpoints,
      'use_checkpoint': cfg.prediccion.use_checkpoint,
      'epochs': EPOCHS
    }
    
   
    y_pred, test_dataloader = predictor(**kwargs)
 
    print("y_pred:", len(y_pred))
    predicciones = pd.DataFrame({'Y': np.zeros(len(y_pred)),
                                 'Ypred': np.zeros(len(y_pred))}).astype('object')
    #print(predicciones)
    import copy
    for it, (_, _, _, y) in enumerate(tqdm((test_dataloader))):
        y_cp = copy.deepcopy(y)
        del y
        predicciones.iloc[it].loc['Y'] = list(np.squeeze(y_cp[:, 1:, :].numpy()))
        predicciones.iloc[it].loc['Ypred'] = list(np.squeeze(y_pred[it]))
    print(len(predicciones))
    
    with open(output, 'wb') as handler:
        pickle.dump(predicciones, handler)
    print(f"Salvando archivo de predicciones en {output}")

if __name__ == "__main__":
    predictZModel()