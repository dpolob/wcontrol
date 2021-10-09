#%%


import numpy as np
import pandas as pd
import torch

from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
    
 
class Seq2SeqDataset(Dataset):
    """
    
    REHACER ENTERO
    Genera el dataset de la forma siguiente:
    Inputs:
        - df: pandas dataframe con los etiquetaXes  
        - pasado: int numero de muestras, sin contar la actual que se sirven
        - futuro: int numero de muestras futuras que se sirven
        - columna: str nombre de la columna de la que tomar los datos
        - columnas_temporales: array<str> nombres de las columnas con etiquetaT temporales
    Output:
    X_f: ndarray (batch, pasado + 1(actual),
    len(columnas_temporales)) De los etiquetaXes que representan el tiempo
    [                       BATCH0                          ]
        T-pasado        T-pasado + 1 (actual)       ...     T0
        [t1,t2,...,tN]  [t1,t2,...,tN]              ...     [t1,t2,...,tN]

    X_etiquetaX: ndarray (batch, pasado + 1(actual), 1) De los etiquetaXes de temperatura, humedad, etc. seleccionado en columna
    [                       BATCH0                          ]
        T-pasado        T-pasado+1       ...     T0
        [V(T-pasado)]   [V(T-pasado+1)]  ...     [V0]
    Y_t: ndarray (batch, fbatchesen(columnas_temporales)) De los etiquetaXes que representan el tiem
                          BATCH0                          ]
                          T+2                    etiquetaF    T+futuro
        [t1,t2,...,tN]  [t1(batchesN]             ...    [t1,t2,...,tN]
        Y_: ndarray (batch,  pasado + 1(actual), 1) De los etiquetaXes de temperatura, humedad, etc seleccionado en columna
                 BATCH0                          ]
        T+1             T+2                 ...     T+futuro
        [V(T+1)]       [V(T+2)]             ...     [V(T+futuro)]
       
    """
    def __init__(self, datasets: list, pasado: int, futuro: int, etiquetaX: str=None, etiquetaF: list=None, etiquetaT: list=None):
        
        assert isinstance(datasets, list), "datasets no es una lista"
        self.datasets = datasets
        self.pasado = pasado
        self.futuro = futuro
        assert isinstance(etiquetaX, str), "etiquetaX no es string"
        self.etiquetaX = etiquetaX
        assert isinstance(etiquetaF, list), "etiquetaF no es list"
        self.etiquetaF = etiquetaF
        assert isinstance(etiquetaT, list), "etiquetaT no es list"
        self.etiquetaT = etiquetaT
        self.longitud = max([_.index.max() for _ in datasets]) - min([_.index.min() for _ in datasets]) + 1

    def __len__(self) -> int:
        return self.longitud - self.futuro - self.pasado

    def __getitem__(self, idx: list) -> torch.tensor:
        batches = len([_ for _ in idx if _ is not None])
        X_f = np.empty(shape=(batches, # batches
                              self.pasado + 1, # sequences
                              len(self.etiquetaF)))  # features 
        X = np.empty(shape=(batches, # batches
                            self.pasado + 1, # sequences
                            1))  # etiquetaX
        Y_t = np.empty(shape=(batches, # batches
                              self.futuro + 1 , # sequences
                              len(self.etiquetaT)))  # etiquetaF
        Y = np.empty(shape=(batches, # batches
                            self.futuro + 1, # sequences
                            1))  # etiquetaX
        batch = 0
        for count, index in enumerate(idx):
            if index is None:
                continue
            else:
                for seq, item in enumerate(range(index - self.pasado, index + 1)):
                    X_f[batch, seq] = self.datasets[count].loc[self.datasets[count].index == item, self.etiquetaF].values
                    X[batch, seq] = self.datasets[count].loc[self.datasets[count].index == item, self.etiquetaX]
                for seq, item in enumerate(range(index, index + self.futuro + 1)):
                    Y_t[batch, seq] = self.datasets[count].loc[self.datasets[count].index == item, self.etiquetaT].values
                    Y[batch, seq] = self.datasets[count].loc[self.datasets[count].index == item, self.etiquetaX]
            batch +=1
        return (torch.from_numpy(X_f).float(),
                torch.from_numpy(X).float(),
                torch.from_numpy(Y_t).float(),
                torch.from_numpy(Y).float())
    
    
# %%
