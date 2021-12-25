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

from common.utils.parser import parser
from common.utils.datasets import dataset_seq2seq as ds
from common.utils.datasets import sampler_seq2seq as sa
from common.utils.trainers import trainerSeq2Seq as tr
from common.utils.loss_functions import lossfunction as lf
#from models.seq2seq import experimento1model as md
import importlib

   
SI = Fore.GREEN + "SI" + Style.RESET_ALL
NO = Fore.RED + "NO" + Style.RESET_ALL

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
    cfg = AttrDict(parser(name, None)(cfg))
    with open(Path(cfg.paths.zmodel.dataset_metadata), 'r') as handler:
        metadata = yaml.safe_load(handler)
    print("Leidos metadatos del dataset")
    
    with open(Path(cfg.paths.zmodel.dataset), 'rb') as handler:
        datasets = pickle.load(handler)
    print(f"Usando {name} como nombre del experimento")
    
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
   
    if 'shuffle' not in cfg.zmodel.dataloaders.train.keys():
        shuffle = True
    else:
        shuffle = cfg.zmodel.dataloaders.train.shuffle
        
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
    if 'shuffle' not in cfg.zmodel.dataloaders.validation.keys():
        shuffle = True
    else:
        shuffle = cfg.zmodel.dataloaders.validation.shuffle
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
    print("\n")

    torch.manual_seed(420)
    np.random.seed(420)

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
                              #additional_metric_fns={"L1_mean_loss": lf.LossFunction(loss='L1', reduction='mean')}
                              save_model=cfg.zmodel.model.save_model,
                              save_model_path=cfg.paths.zmodel.model,
                              early_stop=cfg.zmodel.model.early_stop,
                              alpha = cfg.zmodel.model.decoder.alpha
                              )

    #trainer.lr_find(train_dataloader, model_optimizer, start_lr=1e-5, end_lr=1e-2, num_iter=500)

    if TRAIN and VALIDATION:
        trainer.train(EPOCHS, train_dataloader, valid_dataloader, resume_only_model=True, resume=True, plot=cfg.zmodel.model.plot_intermediate_results)
    else:
        trainer.train(EPOCHS, train_dataloader  , resume_only_model=True, resume=True, plot=cfg.zmodel.model.plot_intermediate_results)
    # for it, (Xt, X, Yt, Y) in enumerate(train_dataloader):
    #     encoder_optimizer.zero_grad()
    #     decoder_optimizer.zero_grad()
    #     print(it, Xt.shape, X.shape, Yt.shape, Y.shape)
    #     ypred = model(Xt, X, Yt, Y)
    #     loss = MSE_mean(ypred,Y)
    #     loss.backward() # Does backpropagation and calculates gradients
    #     encoder_optimizer.step() # Updates the weights accordingly
    #     decoder_optimizer.step()
    #     if it==85:
    #         print(it, ": ", loss.item() )
    #         print("Xt:", np.any(np.isnan(Xt.numpy())))
    #         print("X:", np.any(np.isnan(X.numpy())))
    #         print("Yt:", np.any(np.isnan(Yt.numpy())))
    #         print("Y:", np.any(np.isnan(Y.numpy())))
    #         print("ypred:", ypred)

    with open(chkpts_path / "valid_losses.pickle", 'rb') as handler:
        valid_losses = pickle.load(handler)
    if valid_losses != {}:
        best_epoch = sorted(valid_losses.items(), key=lambda x:x[1])[0][0]
        print(f"Mejor loss de validacion: {best_epoch}")

@cli.command()
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
def pmodel(file):
    
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
    except:
        print(f"{file} no existe o no existe el archivo de metadatos o no existe el archivo de datos de estaciones")
        exit()

    runs_path = Path(cfg.paths.pmodel.runs)
    runs_path.mkdir(parents=True, exist_ok=True)
    print(f"Usando {runs_path} como ruta para runs")
    
    chkpts_path = Path(cfg.paths.zmodel.checkpoints)
    chkpts_path.mkdir(parents=True, exist_ok=True)
    print(f"Usando {chkpts_path} como ruta para guardar checkpoints")
    
    # Parte general de variables     
    device = 'cuda' if cfg.pmodel.model.use_cuda else 'cpu'
    PASADO = cfg.pasado
    FUTURO = cfg.futuro
    Fout = list(cfg.prediccion)  # etiquetas a predecir len = Fout

    # Parte especifica temperatura
    EPOCHS = cfg.pmodel.model.temperatura.epochs
    kwargs_loss = cfg.pmodel.model.temperatura.loss_function
    
    # Predecir datos de la estacion con modelo zmodel
    kwargs_dataloader = {
        'datasets': datasets,
        'fecha_inicio_test': datetime.strptime(cfg.pmodel.dataloaders.train.fecha_inicio, "%Y-%m-%d %H:%M:%S"),
        'fecha_fin_test': datetime.strptime(cfg.pmodel.dataloaders.train.fecha_fin, "%Y-%m-%d %H:%M:%S"),
        'fecha_inicio': datetime.strptime(metadata['fecha_min'], "%Y-%m-%d %H:%M:%S"),
        'pasado': cfg.pasado,
        'futuro': cfg.futuro,
        'etiquetaX': list(cfg.prediccion),
        'etiquetaF': list(cfg.zmodel.model.encoder.features),
        'etiquetaT': list(cfg.zmodel.model.decoder.features),
        'etiquetaP': list(cfg.zmodel.model.decoder.nwp),
        'indice_max': metadata['indice_max'],
        'indice_min': metadata['indice_min']
        }


    kwargs_prediccion= {
        'name': name,
        'device': 'cuda' if cfg.zmodel.model.use_cuda else 'cpu',
        'path_checkpoints': cfg.paths.zmodel.checkpoints,
        'use_checkpoint': cfg.zmodel.dataloaders.test.use_checkpoint,
        'path_model' : cfg.paths.zmodel.model,
        }
 
    test_dataloader = predictor.generar_test_dataset(**kwargs_dataloader)
    for i, a in enumerate(tqdm(test_dataloader)):
        pass
    print(i)
    # y_pred = predictor.predict(test_dataloader, **kwargs_prediccion)  # y_pred = (len(test), N, Ly, Fout)
    
    # assert y_pred.shape[0]==len(test_dataloader), "Revisar y_pred y_pred.shape[0]!!!"
    # assert y_pred.shape[3]==len(list(cfg.prediccion)), "Revisar y_pred.shape[3]!!!"
    # assert y_pred.shape[2]==cfg.futuro, "Revisar y_pred.shape[2]!!!"
   
    # # Creamos la matriz y de salida real, con el mismo shape que las predicciones
    # y_real = np.empty_like(y_pred)
    # y_nwp = np.empty_like(y_pred)
    # for i, (_, _, _, Y, P) in enumerate(tqdm(test_dataloader)):
    #     # hay que quitarles la componente 0 y pasarlos a numpy
    #     y_real[i, ...] = Y[:, 1:, :].numpy()
    #     y_nwp[i, ...] = P[:, 1:, :].numpy()
        
    # predicciones = {'y_pred': y_pred, 'y_real': y_real, 'y_nwp': y_nwp}
   
    # output = Path(cfg.paths.zmodel.predictions)
    # output.parent.mkdir(parents=True, exist_ok=True)
    # print(f"Usando {output} como ruta para guardar predicciones")
    # with open(output, 'wb') as handler:
    #     pickle.dump(predicciones, handler)
    # print(f"Salvando archivo de predicciones en {output}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # TRAIN = cfg.pmodel.dataloaders.train.enable
    # if TRAIN:
    #     FECHA_INICIO_TRAIN = datetime.strptime(cfg.pmodel.dataloaders.train.fecha_inicio, "%Y-%m-%d %H:%M:%S")
    #     FECHA_FIN_TRAIN = datetime.strptime(cfg.pmodel.dataloaders.train.fecha_fin, "%Y-%m-%d %H:%M:%S")
    # else:
    #     FECHA_INICIO_TRAIN = None
    #     FECHA_FIN_TRAIN = None
    
    # VALIDATION = cfg.pmodel.dataloaders.validation.enable
    # if VALIDATION:
    #     FECHA_INICIO_VALID = datetime.strptime(cfg.pmodel.dataloaders.validation.fecha_inicio, "%Y-%m-%d %H:%M:%S")
    #     FECHA_FIN_VALID = datetime.strptime(cfg.pmodel.dataloaders.validation.fecha_fin, "%Y-%m-%d %H:%M:%S")
    # else:
    #     FECHA_INICIO_VALID = None
    #     FECHA_FIN_VALID = None
        

    # print(f"Train: {SI if TRAIN else NO}, Validation: {SI if VALIDATION else NO}\n")
    
    # fecha_inicio = datetime.strptime(metadata['fecha_min'], "%Y-%m-%d %H:%M:%S")
       
    # # Split datasets
    # if TRAIN:
    #     x = (FECHA_INICIO_TRAIN - fecha_inicio).days * 24 + (FECHA_INICIO_TRAIN - fecha_inicio).seconds / 3600
    #     y = (FECHA_FIN_TRAIN - fecha_inicio).days * 24 + (FECHA_FIN_TRAIN - fecha_inicio).seconds / 3600
    #     # x=0; y=500
    #     print(f"Generando dataset de train desde {x} a {y}")
    #     dfs_train = [df.loc[(df.index >= x ) & (df.index <= y), :] for df in datasets]
    # if VALIDATION:
    #     x = (FECHA_INICIO_VALID - fecha_inicio).days * 24 + (FECHA_INICIO_VALID - fecha_inicio).seconds / 3600
    #     y = (FECHA_FIN_VALID - fecha_inicio).days * 24 + (FECHA_FIN_VALID - fecha_inicio).seconds / 3600
    #     # x=1001; y=1500
    #     print(f"Generando dataset de validation desde {x} a {y}")
    #     dfs_valid = [df.loc[(df.index >= x ) & (df.index <= y), :] for df in datasets]
   
    # if 'shuffle' not in cfg.zmodel.dataloaders.train.keys():
    #     shuffle = True
    # else:
    #     shuffle = cfg.zmodel.dataloaders.train.shuffle
        
    # if TRAIN:
    #     train_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(datasets=dfs_train,
    #                                                             pasado=PASADO,
    #                                                             futuro=FUTURO,
    #                                                             etiquetaX=Fout,
    #                                                             etiquetaF=Ff,
    #                                                             etiquetaT=Ft,
    #                                                             etiquetaP=Fnwp),
    #                                   sampler=sa.Seq2SeqSampler(datasets=dfs_train,
    #                                                             pasado=PASADO,
    #                                                             futuro=FUTURO,
    #                                                             shuffle=shuffle),
    
    #                                   batch_size=None,
    #                                   num_workers=8)
    # if 'shuffle' not in cfg.zmodel.dataloaders.validation.keys():
    #     shuffle = True
    # else:
    #     shuffle = cfg.zmodel.dataloaders.validation.shuffle
    # if VALIDATION:
    #     valid_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(datasets=dfs_valid,
    #                                                             pasado=PASADO,
    #                                                             futuro=FUTURO,
    #                                                             etiquetaX=Fout,
    #                                                             etiquetaF=Ff,
    #                                                             etiquetaT=Ft,
    #                                                             etiquetaP=Fnwp),
    #                                   sampler=sa.Seq2SeqSampler(datasets=dfs_valid,
    #                                                             pasado=PASADO,
    #                                                             futuro=FUTURO,
    #                                                             shuffle=shuffle),
    
    #                                   batch_size=None,
    #                                   num_workers=8)
   

    # print(f"Dataset de train: {len(dfs_train)} componentes con longitud máxima de {len(train_dataloader)}")
    # print(f"Dataset de validacion: {len(dfs_valid)} componentes con longitud máxima de {len(valid_dataloader)}")
    # print("\n")

    # torch.manual_seed(420)
    # np.random.seed(420)

    # module = importlib.import_module(f"models.seq2seq.{cfg.zmodel.model.name}")
    
    # encoder = module.RNNEncoder(rnn_num_layers=cfg.zmodel.model.encoder.rnn_num_layers,
    #                             input_feature_len=len(Ff),
    #                             sequence_len=PASADO + 1, # +1 porque tomamos tiempo0
    #                             hidden_size=cfg.zmodel.model.encoder.hidden_size,
    #                             bidirectional=cfg.zmodel.model.encoder.bidirectional,
    #                             device=device,
    #                             rnn_dropout=cfg.zmodel.model.encoder.rnn_dropout)
    # encoder = encoder.to(device)

    # decoder = module.DecoderCell(input_feature_len=len(Ft) + len(Fnwp),
    #                              hidden_size=cfg.zmodel.model.decoder.hidden_size,
    #                              output_size=len(Fout),
    #                              dropout=cfg.zmodel.model.decoder.dropout)
    # decoder = decoder.to(device)

    # model = module.EncoderDecoderWrapper(encoder=encoder,
    #                                 decoder_cell=decoder,
    #                                 output_size=len(Fout),
    #                                 output_sequence_len=FUTURO,
    #                                 device=device)
    # model = model.to(device)


    # # Funciones loss
    # loss_fn = lf.LossFunction(**kwargs_loss)
    
    # encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.zmodel.model.encoder.lr, weight_decay=cfg.zmodel.model.decoder.lr / 10)
    # decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=cfg.zmodel.model.decoder.lr, weight_decay=cfg.zmodel.model.decoder.lr / 10)
    # if cfg.zmodel.model.scheduler:
    #     encoder_scheduler = optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=cfg.zmodel.model.encoder.lr, steps_per_epoch=len(train_dataloader), epochs=EPOCHS)
    #     decoder_scheduler = optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=cfg.zmodel.model.decoder.lr, steps_per_epoch=len(train_dataloader), epochs=EPOCHS)
    #     scheduler = [encoder_scheduler, decoder_scheduler]
    # else:
    #     scheduler = None
    # model_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.zmodel.model.decoder.lr, weight_decay=cfg.zmodel.model.decoder.lr / 10)
    # scheduler = optim.lr_scheduler.OneCycleLR(model_optimizer, max_lr=cfg.zmodel.model.decoder.lr, steps_per_epoch=len(train_dataloader), epochs=6)

    
    # trainer = tr.TorchTrainer(name=name,
    #                           model=model,
    #                           optimizer=[encoder_optimizer, decoder_optimizer], 
    #                           loss_fn = loss_fn,   
    #                           scheduler=None,
    #                           device=device,
    #                           scheduler_batch_step=True if cfg.zmodel.model.scheduler else False,
    #                           checkpoint_folder= chkpts_path,
    #                           runs_folder= runs_path,
    #                           #additional_metric_fns={"L1_mean_loss": lf.LossFunction(loss='L1', reduction='mean')}
    #                           save_model=cfg.zmodel.model.save_model,
    #                           save_model_path=cfg.paths.zmodel.model,
    #                           early_stop=cfg.zmodel.model.early_stop,
    #                           alpha = cfg.zmodel.model.decoder.alpha
    #                           )

    # #trainer.lr_find(train_dataloader, model_optimizer, start_lr=1e-5, end_lr=1e-2, num_iter=500)

    # if TRAIN and VALIDATION:
    #     trainer.train(EPOCHS, train_dataloader, valid_dataloader, resume_only_model=True, resume=True, plot=cfg.zmodel.model.plot_intermediate_results)
    # else:
    #     trainer.train(EPOCHS, train_dataloader  , resume_only_model=True, resume=True, plot=cfg.zmodel.model.plot_intermediate_results)
    # # for it, (Xt, X, Yt, Y) in enumerate(train_dataloader):
    # #     encoder_optimizer.zero_grad()
    # #     decoder_optimizer.zero_grad()
    # #     print(it, Xt.shape, X.shape, Yt.shape, Y.shape)
    # #     ypred = model(Xt, X, Yt, Y)
    # #     loss = MSE_mean(ypred,Y)
    # #     loss.backward() # Does backpropagation and calculates gradients
    # #     encoder_optimizer.step() # Updates the weights accordingly
    # #     decoder_optimizer.step()
    # #     if it==85:
    # #         print(it, ": ", loss.item() )
    # #         print("Xt:", np.any(np.isnan(Xt.numpy())))
    # #         print("X:", np.any(np.isnan(X.numpy())))
    # #         print("Yt:", np.any(np.isnan(Yt.numpy())))
    # #         print("Y:", np.any(np.isnan(Y.numpy())))
    # #         print("ypred:", ypred)

    # with open(chkpts_path / "valid_losses.pickle", 'rb') as handler:
    #     valid_losses = pickle.load(handler)
    # if valid_losses != {}:
    #     best_epoch = sorted(valid_losses.items(), key=lambda x:x[1])[0][0]
    #     print(f"Mejor loss de validacion: {best_epoch}")
if __name__ == "__main__":
    cli()
