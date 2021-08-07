# Dataloader y Sampler

import numpy as np
#import pandas as pd
import torch

from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset


class Seq2SeqSampler(Sampler):
    """
    Input:
        dataset (DataFrame): Datos de entrada
        pasado (int): numero de muestras, sin contar la actual que se sirven
        futuro (int): numero de muestras futuras que se sirven
        batch_size (int): tamaño del batch a servir
        shuffle (bool): muestras aleatorias o secuenciales
    Output:
        out (array<int>): indices del DataFrame
    """
    def __init__(self, dataset, pasado, futuro, batch_size, shuffle):
        self.dataset = dataset
        self.pasado = pasado
        self.futuro = futuro
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.i = 0
      
    def __iter__(self):
        self.vector = list(range(self.pasado, len(self.dataset) - self.futuro, 1))
        if self.shuffle:
            np.random.shuffle(self.vector)
        self.i = 0
        return(self)
    
    def __next__(self):
        if self.i >= len(self.vector):
            raise StopIteration
        else:
            i = self.i
            if (i + self.batch_size) < len(self.vector):
                self.i += self.batch_size
                return [self.vector[_] for _ in range(i, i + self.batch_size)]
            else:
                self.i += self.batch_size
                return [self.vector[_] for _ in range(i, len(self.vector))]
    
    def __len__(self):
        return len(self.dataset) - self.futuro - self.pasado
  

    
 
class Seq2SeqDataset(Dataset):
    """
    Genera el dataset de la forma siguiente:
    Inputs:
        - df: pandas dataframe con los valores  
        - pasado: int numero de muestras, sin contar la actual que se sirven
        - futuro: int numero de muestras futuras que se sirven
        - columna: str nombre de la columna de la que tomar los datos
        - columnas_temporales: array<str> nombres de las columnas con variables temporales
    Output:
    X_tiempo: ndarray (batch, pasado + 1(actual), len(columnas_temporales)) De los valores que representan el tiempo
    [                       BATCH0                          ]
        T-pasado        T-pasado + 1 (actual)       ...     T0
        [t1,t2,...,tN]  [t1,t2,...,tN]              ...     [t1,t2,...,tN]

    X_valor: ndarray (batch, pasado + 1(actual), 1) De los valores de temperatura, humedad, etc. seleccionado en columna
    [                       BATCH0                          ]
        T-pasado        T-pasado+1       ...     T0
        [V(T-pasado)]   [V(T-pasado+1)]  ...     [V0]

    Y_tiempo: ndarray (batch, futuro, len(columnas_temporales)) De los valores que representan el tiempo
    [                       BATCH0                          ]
        T+1             T+2                        ...     T+futuro
        [t1,t2,...,tN]  [t1,t2,...,tN]             ...    [t1,t2,...,tN]

    Y_: ndarray (batch, pasado + 1(actual), 1) De los valores de temperatura, humedad, etc seleccionado en columna
    [                       BATCH0                          ]
        T+1             T+2                 ...     T+futuro
        [V(T+1)]       [V(T+2)]             ...     [V(T+futuro)]
       
    """
    def __init__(self, dataset, pasado, futuro, columna_valor=None, columna_tiempo=None):
        self.df = dataset
        self.pasado = pasado
        self.futuro = futuro
        self.columna_valor = columna_valor
        self.columna_tiempo = columna_tiempo

        assert isinstance(self.columna_valor, str), "columna_valor no es string"
        assert isinstance(self.columna_tiempo, list), "columna_tiempo no es list"
        assert self.columna_valor in self.df.keys(), "columna_valor no existe en el dataset"


    def __len__(self):
        return len(self.df) - self.futuro - self.pasado

    def __getitem__(self, idx):
        X_tiempo = np.empty(shape=(len(idx), # batches
                                self.pasado + 1, # sequences
                                len(self.columna_tiempo)))  # features de tiempo
        X = np.empty(shape=(len(idx), # batches
                        self.pasado + 1, # sequences
                        1))  # valor

        Y_tiempo = np.empty(shape=(len(idx), # batches
                        self.futuro, # sequences
                        len(self.columna_tiempo)))  # features
        Y = np.empty(shape=(len(idx), # batches
                        self.futuro, # sequences
                        1))  # valor
        for batch, index in enumerate(idx):
            for seq, item in enumerate(range(index - self.pasado, index + 1)):
                X_tiempo[batch, seq] = self.df.iloc[item].loc[self.columna_tiempo].values
                X[batch, seq] = self.df.iloc[item].loc[self.columna_valor]
            for seq, item in enumerate(range(index +1, index + self.futuro +1)):
                Y_tiempo[batch, seq] = self.df.iloc[item].loc[self.columna_tiempo].values
                Y[batch, seq] = self.df.iloc[item].loc[self.columna_valor]
        
        return torch.from_numpy(X_tiempo).float(), torch.from_numpy(X).float(), \
               torch.from_numpy(Y_tiempo).float(), torch.from_numpy(Y).float()
    