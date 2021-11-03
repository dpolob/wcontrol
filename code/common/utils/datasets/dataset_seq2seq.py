#%%


import numpy as np
import pandas as pd
import torch

from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
    
 
class Seq2SeqDataset(Dataset):
    """
    
    REHACER ENTERO
        
    """
    def __init__(self, datasets: list, pasado: int, futuro: int, etiquetaX: list=None, etiquetaF: list=None, etiquetaT: list=None, etiquetaP: list=None) -> tuple:
        
        assert isinstance(datasets, list), "datasets no es una lista"
        self.datasets = datasets
        self.pasado = pasado
        self.futuro = futuro
        assert isinstance(etiquetaX, list), "etiquetaX no es list"
        self.etiquetaX = etiquetaX
        assert isinstance(etiquetaF, list), "etiquetaF no es list"
        self.etiquetaF = etiquetaF
        assert isinstance(etiquetaT, list), "etiquetaT no es list"
        self.etiquetaT = etiquetaT
        assert isinstance(etiquetaP, list), "etiquetaP no es list"
        self.etiquetaP = etiquetaP
        self.longitud = max([_.index.max() for _ in datasets]) - min([_.index.min() for _ in datasets]) + 1

    def __len__(self) -> int:
        return self.longitud - self.futuro - self.pasado

    def __getitem__(self, idx: list) -> torch.tensor:
        batches = len([_ for _ in idx if _ is not None])
        X_f = np.empty(shape=(batches, # batches
                              self.pasado + 1, # sequences
                              len(self.etiquetaF)))  # features 
        X = np.empty(shape=(batches, # batches
                            self.pasado + 1, # sequences + 1 [-pasado...0] 
                            len(self.etiquetaX)))  # etiquetaX
        Y_t = np.empty(shape=(batches, # batches
                              self.futuro + 1 , # sequences
                              len(self.etiquetaT)))  # etiquetaF
        Y = np.empty(shape=(batches, # batches
                            self.futuro + 1, # sequences + 1
                            len(self.etiquetaX)))  # etiquetaX
        P = np.empty(shape=(batches, # batches
                            self.futuro + 1, # sequences + 1 
                            len(self.etiquetaP)))  # etiquetaP
        batch = 0
        for count, index in enumerate(idx):
            if index is None:
                continue
            else:
                for seq, item in enumerate(range(index - self.pasado, index + 1)):
                    X_f[batch, seq] = self.datasets[count].loc[self.datasets[count].index == item, self.etiquetaF].values
                    X[batch, seq] = self.datasets[count].loc[self.datasets[count].index == item, self.etiquetaX].values
                for seq, item in enumerate(range(index, index + self.futuro + 1)):
                    Y_t[batch, seq] = self.datasets[count].loc[self.datasets[count].index == item, self.etiquetaT].values
                    Y[batch, seq] = self.datasets[count].loc[self.datasets[count].index == item, self.etiquetaX].values
                    P[batch, seq] = self.datasets[count].loc[self.datasets[count].index == item, self.etiquetaP].values
            batch +=1
        return (torch.from_numpy(X_f).float(),  # X_f
                torch.from_numpy(X).float(),  # X
                torch.from_numpy(Y_t).float(),  # Y_t
                torch.from_numpy(Y).float(),  # Y
                torch.from_numpy(P).float())  # P  
    
    
# %%
