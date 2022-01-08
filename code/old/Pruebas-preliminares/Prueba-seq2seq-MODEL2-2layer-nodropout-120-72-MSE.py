#%%
# -*- coding: utf-8 -*-
# Preprocesado
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm

datos = Path('/home/diego/weather-control/data/raw/agoncillo-temp-20180803-20210803/datos.csv')
df = pd.read_csv(datos, sep=';', decimal=',', header='infer')
df['Fecha'] = pd.to_datetime(df['Fecha'], format="%Y-%m-%d %H:%M")
df.set_index(df['Fecha'], drop=True, inplace=True)


# cada hora
df = df.resample('1H').mean()

# calcular senos y cosenos
df['Fecha'] = df.index

df['day'] = df['Fecha'].apply(lambda x: x.day)
df['day_sin'] = np.sin(df['day'])
df['day_cos'] = np.cos(df['day'])

df['dayofweek'] = df['Fecha'].apply(lambda x: x.isoweekday())
df['dayofweek_sin'] = np.sin(df['dayofweek'])
df['dayofweek_cos'] = np.cos(df['dayofweek'])

df['month'] = df['Fecha'].apply(lambda x: x.month)
df['month_sin'] = np.sin(df['month'])
df['month_cos'] = np.cos(df['month'])

df['year'] = df['Fecha'].apply(lambda x: x.year)
df['year_sin'] = np.sin(df['year'])
df['year_cos'] = np.cos(df['year'])

df.drop(columns=['Fecha', 'day', 'dayofweek', 'month', 'year'], inplace=True)

# escalado  no uso minmaxscaler
def scaler(X, Xmin=-20, Xmax=50, min=0, max=1):
    """
    Transform features by scaling each feature to a given range.
    This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.
    where min, max = feature_range.
    This transformation is often used as an alternative to zero mean, unit variance scaling.
    """
    
    X_std = (X - Xmin) / (Xmax - Xmin)
    X = X_std * (max - min) + min
    return X

df['Temperatura'] = df['Temperatura'].apply(scaler)
# quitar nan
df.dropna(inplace=True)
df_test = df.head(10)


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from common.utils.datasets import dataset_seq2seq as ds
from common.utils.trainers import trainer_seq2seq as tr
from models.seq2seq import model_doscapas as md

TRAIN = False
PASADO = 5 * 24 
FUTURO = 3 * 24
BATCH_SIZE = 1
EPOCHS = 40
NAME = f'seq2seq-MODEL2-2layer-nodropout-{PASADO}-{FUTURO}-MSE'
device = 'cuda'
COLUMNA_VALOR = 'Temperatura'
COLUMNA_TIEMPO = ['day_sin', 'day_cos', 'dayofweek_sin', 'dayofweek_cos',
                  'month_sin', 'month_cos', 'year_sin', 'year_cos']

FECHA_FIN_TRAIN = '2020-12-31'
FECHA_FIN_VALID = '2021-04-30'

# Split dataset
#df_train = df[df.index <= '2019-12-31']
df_train = df[(df.index <= FECHA_FIN_TRAIN)]
df_valid = df[(df.index > FECHA_FIN_TRAIN) & (df.index <= FECHA_FIN_VALID)]
df_test = df[df.index > FECHA_FIN_VALID]

train_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(dataset=df_train,
                                                        pasado=PASADO,
                                                        futuro=FUTURO,
                                                        columna_tiempo=COLUMNA_TIEMPO,
                                                        columna_valor=COLUMNA_VALOR),
                              sampler=ds.Seq2SeqSampler(dataset=df_train,
                                                        futuro=FUTURO,
                                                        pasado=PASADO,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True),
                              batch_size=None,
                              num_workers=8)

valid_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(dataset=df_valid,
                                                        pasado=PASADO,
                                                        futuro=FUTURO,
                                                        columna_tiempo=COLUMNA_TIEMPO,
                                                        columna_valor=COLUMNA_VALOR),
                                sampler=ds.Seq2SeqSampler(dataset=df_valid,
                                                        futuro=FUTURO,
                                                        pasado=PASADO,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False),
                                batch_size=None,
                                num_workers=8)

test_dataloader = DataLoader(dataset=ds.Seq2SeqDataset(dataset=df_test,
                                                        pasado=PASADO,
                                                        futuro=FUTURO,
                                                        columna_tiempo=COLUMNA_TIEMPO,
                                                        columna_valor=COLUMNA_VALOR),
                                sampler=ds.Seq2SeqSampler(dataset=df_test,
                                                        futuro=FUTURO,
                                                        pasado=PASADO,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False),
                                batch_size=None,
                                num_workers=8)
print(NAME + "TRAIN " + str(TRAIN))
print("Dataset de train: ", len(train_dataloader))
print("Dataset de validacion: ", len(valid_dataloader))
print("Dataset de test", len(test_dataloader))


torch.manual_seed(420)
np.random.seed(420)

encoder = md.RNNEncoder(rnn_num_layers=2,
                     input_feature_len=len(COLUMNA_TIEMPO) + 1,
                     sequence_len=PASADO + 1,
                     hidden_size=200,
                     bidirectional=False,
                     device=device,
                     rnn_dropout=0.0)
encoder = encoder.to(device)

decoder = md.DecoderCell(input_feature_len=len(COLUMNA_TIEMPO) + 1,
                      hidden_size=200,
                      dropout=0.2)
decoder = decoder.to(device)

model = md.EncoderDecoderWrapper(encoder=encoder,
                              decoder_cell=decoder,
                              output_size=1,
                              output_sequence_len=FUTURO,
                              teacher_forcing=0.3,
                              device=device)
model = model.to(device)

#loss_function = nn.MSELoss(reduction='mean')
class LossFunction():
    def __init__(self):
        self.lf1 = nn.MSELoss(reduction='mean')
        #self.lf2 = nn.L1Loss(reduction='none')
    def __call__(self, y_pred, y):
        return self.lf1(y_pred, y)

loss_function = LossFunction()
encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=1e-3)
decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=1e-3)

encoder_scheduler = optim.lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=1e-4, steps_per_epoch=len(train_dataloader), epochs=EPOCHS)
decoder_scheduler = optim.lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=1e-4, steps_per_epoch=len(train_dataloader), epochs=EPOCHS)

model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
#scheduler = optim.lr_scheduler.OneCycleLR(model_optimizer, max_lr=3e-3, steps_per_epoch=len(train_dataloader), epochs=6)


trainer = tr.TorchTrainer(NAME,
                        model,
                       [encoder_optimizer, decoder_optimizer],
                       loss_function,   
                       [encoder_scheduler, decoder_scheduler],
                       device,
                       scheduler_batch_step=True,
                       pass_y=True,
                       checkpoint_folder= f'/home/diego/weather-control/experiments/modelchkpts/{NAME}_chkpts',
                       runs_folder= f'/home/diego/weather-control/experiments/runs/{NAME}'
                       #additional_metric_fns={'SMAPE': smape_exp_loss}
                       )


#trainer.lr_find(train_dataloader, model_optimizer, start_lr=1e-5, end_lr=1e-2, num_iter=500)
if TRAIN:
    trainer.train(EPOCHS, train_dataloader, valid_dataloader, resume_only_model=True, resume=True)

# for it, (Xt, X, Yt, Y) in enumerate(train_dataloader):
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
    
#     ypred = model(Xt, X, Yt, Y)
#     loss = loss_function(ypred,Y)
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

# # cargar mejor checkpoint
trainer._load_checkpoint(epoch=35)
y_pred = trainer.predict(test_dataloader)  # y_pred (len(test), N, L, F(d)out) (4000,1,72,1)

print(len(y_pred))
predicciones = pd.DataFrame({'Y': np.zeros(len(y_pred)),
                             'Ypred': np.zeros(len(y_pred))}).astype('object')

for it, (xt, x, yt, y) in enumerate(tqdm((test_dataloader))):
    predicciones.iloc[it].loc['Y'] = list(np.squeeze(y.numpy()))
    predicciones.iloc[it].loc['Ypred'] = list(np.squeeze(y_pred[it]))
#%%
import matplotlib.pyplot as plt

plt.plot(predicciones.iloc[501].loc['Y'], 'b')
plt.plot(predicciones.iloc[501].loc['Ypred'], 'r')



# %%
