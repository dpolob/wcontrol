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

import common.utils.parser as parser
from common.utils.datasets import dataset_seq2seq as ds
from common.utils.datasets import sampler_seq2seq as sa
from common.utils.trainers import trainerSeq2Seq as tr
from common.utils.loss_functions import lossfunction as lf
#from models.seq2seq import experimento1model as md
import importlib

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
    cfg = AttrDict(parser(name, _)(cfg))
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
    PREDICCION = list(cfg.prediccion)
    FEATURES = list(cfg.zmodel.model.encoder.features)
    DEFINIDAS = list(cfg.zmodel.model.decoder.features)
    NWP = list(cfg.zmodel.model.decoder.nwp)
    OUTPUT_SIZE = len(PREDICCION)
    
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
        
   
    SI = Fore.GREEN + "SI" + Style.RESET_ALL
    NO = Fore.RED + "NO" + Style.RESET_ALL
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
                                                                etiquetaX=PREDICCION,
                                                                etiquetaF=FEATURES,
                                                                etiquetaT=DEFINIDAS,
                                                                etiquetaP=NWP),
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
                                                                etiquetaX=PREDICCION,
                                                                etiquetaF=FEATURES,
                                                                etiquetaT=DEFINIDAS,
                                                                etiquetaP=NWP),
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
                            input_feature_len=len(FEATURES) + 1,
                            sequence_len=PASADO + 1,
                            hidden_size=cfg.zmodel.model.encoder.hidden_size,
                            bidirectional=cfg.zmodel.model.encoder.bidirectional,
                            device=device,
                            rnn_dropout=cfg.zmodel.model.encoder.rnn_dropout)
    encoder = encoder.to(device)

    decoder = module.DecoderCell(input_feature_len=len(DEFINIDAS) + 1,
                                 hidden_size=cfg.zmodel.model.decoder.hidden_size,
                                 output_size=len(PREDICCION),
                                 dropout=cfg.zmodel.model.decoder.dropout)
    decoder = decoder.to(device)

    model = module.EncoderDecoderWrapper(encoder=encoder,
                                    decoder_cell=decoder,
                                    output_size=OUTPUT_SIZE,
                                    output_sequence_len=FUTURO,
                                    teacher_forcing=cfg.zmodel.model.teacher_forcing,
                                    duplicate_teaching=cfg.zmodel.model.duplicate_teaching,
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
                              pass_y=True,
                              checkpoint_folder= chkpts_path,
                              runs_folder= runs_path,
                              #additional_metric_fns={"L1_mean_loss": lf.LossFunction(loss='L1', reduction='mean')}
                              save_model=cfg.zmodel.model.save_model,
                              save_model_path=cfg.paths.zmodel.model,
                              early_stop=cfg.zmodel.model.early_stop
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

if __name__ == "__main__":
    cli()
