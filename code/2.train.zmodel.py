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

from common.utils.datasets import experimento1dataset as ds
from common.utils.datasets import experimento1sampler as sa
from common.utils.trainers import experimento1trainer as tr
from common.utils.loss_functions import lossfunction as lf
#from models.seq2seq import experimento1model as md
import importlib


@click.command()
@click.option('--file', type=click.Path(exists=True), help='path/to/.yml Ruta al archivo de configuracion')
def entrenamientoZModel(file):
    
    try:
        with open(file, 'r') as handler:
            cfg = AttrDict(yaml.safe_load(handler))
        print(f"Usando {file} como archivo de configuracion")
    except:
        print(f"{file} no existe. Por favor defina un archivo con --file")
        exit()
    
    with open(Path(cfg.paths.datasets), 'rb') as handler:
        datasets = pickle.load(handler)

    name = cfg.experiment
    print(f"Usando {name} como nombre del experimento")
    
    runs_path = Path(cfg.paths.runs) / name            
    runs_path.mkdir(parents=True, exist_ok=True)
    print(f"Usando {runs_path} como ruta para runs")
    
    chkpts_path = Path(cfg.paths.checkpoints) / name
    chkpts_path.mkdir(parents=True, exist_ok=True)
    print(f"Usando {chkpts_path} como ruta para guardar checkpoints")
         
    device = 'cuda' if cfg.entrenamiento.use_cuda else 'cpu'
    PASADO = cfg.entrenamiento.pasado
    FUTURO = cfg.entrenamiento.futuro
    EPOCHS = cfg.entrenamiento.epochs
    kwargs_loss = cfg.entrenamiento.loss_function
    PREDICCION = cfg.entrenamiento.prediccion
    FEATURES = list(cfg.entrenamiento.features)
    DEFINIDAS = list(cfg.entrenamiento.definidas)
    
    TRAIN = cfg.entrenamiento.dataloaders.train.enable
    if TRAIN:
        FECHA_INICIO_TRAIN = datetime.strptime(cfg.entrenamiento.dataloaders.train.fecha_inicio, "%Y-%m-%d %H:%M:%S")
        FECHA_FIN_TRAIN = datetime.strptime(cfg.entrenamiento.dataloaders.train.fecha_fin, "%Y-%m-%d %H:%M:%S")
    else:
        FECHA_INICIO_TRAIN = None
        FECHA_FIN_TRAIN = None
    
    VALIDATION = cfg.entrenamiento.dataloaders.validation.enable
    if VALIDATION:
        FECHA_INICIO_VALID = datetime.strptime(cfg.entrenamiento.dataloaders.validation.fecha_inicio, "%Y-%m-%d %H:%M:%S")
        FECHA_FIN_VALID = datetime.strptime(cfg.entrenamiento.dataloaders.validation.fecha_fin, "%Y-%m-%d %H:%M:%S")
    else:
        FECHA_INICIO_VALID = None
        FECHA_FIN_VALID = None
        
   
    SI = Fore.GREEN + "SI" + Style.RESET_ALL
    NO = Fore.RED + "NO" + Style.RESET_ALL
    print(f"Train: {SI if TRAIN else NO}, Validation: {SI if VALIDATION else NO}\n")
    
    inicio = min([_ for _ in [FECHA_INICIO_TRAIN, FECHA_INICIO_VALID] if _ is not None])
    
    # Split datasets
    if TRAIN:
        x = (FECHA_INICIO_TRAIN - inicio).days * 24 + (FECHA_INICIO_TRAIN - inicio).seconds / 3600
        y = (FECHA_FIN_TRAIN - inicio).days * 24 + (FECHA_FIN_TRAIN - inicio).seconds / 3600
        # x=0; y=500
        print(f"Generando dataset de train desde {x} a {y}")
        dfs_train = [df.loc[(df.index >= x ) & (df.index <= y), :] for df in datasets]
    if VALIDATION:
        x = (FECHA_INICIO_VALID - inicio).days * 24 + (FECHA_INICIO_VALID - inicio).seconds / 3600
        y = (FECHA_FIN_VALID - inicio).days * 24 + (FECHA_FIN_VALID - inicio).seconds / 3600
        # x=1001; y=1500
        print(f"Generando dataset de validation desde {x} a {y}")
        dfs_valid = [df.loc[(df.index >= x ) & (df.index <= y), :] for df in datasets]
   
    if TRAIN:
        train_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(datasets=dfs_train,
                                                                pasado=PASADO,
                                                                futuro=FUTURO,
                                                                etiquetaX=PREDICCION,
                                                                etiquetaF=FEATURES,
                                                                etiquetaT=DEFINIDAS),
                                      sampler=sa.Seq2SeqSampler(datasets=dfs_train,
                                                                pasado=PASADO,
                                                                futuro=FUTURO,
                                                                shuffle=True),
    
                                      batch_size=None,
                                      num_workers=8)
    if VALIDATION:
        valid_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(datasets=dfs_valid,
                                                                pasado=PASADO,
                                                                futuro=FUTURO,
                                                                etiquetaX=PREDICCION,
                                                                etiquetaF=FEATURES,
                                                                etiquetaT=DEFINIDAS),
                                      sampler=sa.Seq2SeqSampler(datasets=dfs_valid,
                                                                pasado=PASADO,
                                                                futuro=FUTURO,
                                                                shuffle=False),
    
                                      batch_size=None,
                                      num_workers=8)
   

    print(f"Dataset de train: {len(dfs_train)} componentes con longitud máxima de {len(train_dataloader)}")
    print(f"Dataset de validacion: {len(dfs_valid)} componentes con longitud máxima de {len(valid_dataloader)}")
    print("\n")

    torch.manual_seed(420)
    np.random.seed(420)

    module = importlib.import_module(f"models.seq2seq.{cfg.entrenamiento.model.name}")
    
    encoder = module.RNNEncoder(rnn_num_layers=cfg.entrenamiento.encoder.rnn_num_layers,
                            input_feature_len=len(FEATURES) + 1,
                            sequence_len=PASADO + 1,
                            hidden_size=cfg.entrenamiento.encoder.hidden_size,
                            bidirectional=cfg.entrenamiento.encoder.bidirectional,
                            device=device,
                            rnn_dropout=cfg.entrenamiento.encoder.rnn_dropout)
    encoder = encoder.to(device)

    decoder = module.DecoderCell(input_feature_len=len(DEFINIDAS) + 1,
                            hidden_size=cfg.entrenamiento.decoder.hidden_size,
                            dropout=cfg.entrenamiento.decoder.dropout)
    decoder = decoder.to(device)

    model = module.EncoderDecoderWrapper(encoder=encoder,
                                    decoder_cell=decoder,
                                    output_size=1,
                                    output_sequence_len=FUTURO,
                                    teacher_forcing=cfg.entrenamiento.model.teacher_forcing,
                                    duplicate_teaching=cfg.entrenamiento.model.duplicate_teaching,
                                    device=device)
    model = model.to(device)


    # Funciones loss
    loss_fn = lf.LossFunction(**kwargs_loss)
    
    encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.entrenamiento.encoder.lr, weight_decay=1e-3)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=cfg.entrenamiento.decoder.lr, weight_decay=1e-3)
    if cfg.entrenamiento.scheduler:
        encoder_scheduler = optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataloader), epochs=EPOCHS)
        decoder_scheduler = optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataloader), epochs=EPOCHS)
        scheduler = [encoder_scheduler, decoder_scheduler]
    else:
        scheduler = None
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(model_optimizer, max_lr=3e-3, steps_per_epoch=len(train_dataloader), epochs=6)

    
    trainer = tr.TorchTrainer(name=name,
                              model=model,
                              optimizer=[encoder_optimizer, decoder_optimizer], 
                              loss_fn = loss_fn,   
                              scheduler=None,
                              device=device,
                              scheduler_batch_step=True if cfg.entrenamiento.scheduler else False,
                              pass_y=True,
                              checkpoint_folder= chkpts_path,
                              runs_folder= runs_path,
                              #additional_metric_fns={"L1_mean_loss": lf.LossFunction(loss='L1', reduction='mean')}
                              save_model=cfg.entrenamiento.save_model
                              )

    #trainer.lr_find(train_dataloader, model_optimizer, start_lr=1e-5, end_lr=1e-2, num_iter=500)

    if TRAIN and VALIDATION:
        trainer.train(EPOCHS, train_dataloader, valid_dataloader, resume_only_model=True, resume=True, plot=cfg.entrenamiento.plot_intermediate_results)
    else:
        trainer.train(EPOCHS, train_dataloader  , resume_only_model=True, resume=True, plot=cfg.entrenamiento.plot_intermediate_results)
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
        valid_losses = pickle(handler)
    if valid_losses != {}:
        best_epoch = sorted(valid_losses.items(), key=lambda x:x[1])[0][0]
        print(f"Mejor loss de validacion: {best_epoch}")

if __name__ == "__main__":
    entrenamientoZModel()
