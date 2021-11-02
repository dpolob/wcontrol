import numpy as np
#import pandas as pd
import torch

from torch.utils.data.sampler import Sampler
#from torch.utils.data import Dataset


class Seq2SeqSampler(Sampler):
    """Implementa un sampler específico
    """
    def __init__(self, datasets: list, pasado: int, futuro: int, shuffle: bool=False) -> list:
        """Devuelve un array con los indices de los muestras a obtener

        Args:
            dataset (list): Lista de pd.DataFrame
            pasado (int): Numero de muestras a tomar en el pasado.
            futuro (int): Numero de muestras a tomar en el futuro.
            shuffle (bool, optional): Duevuelve desordenadas las muestras. Defaults to False.

        Returns:
            list: Lista de indices de los pd.DataFrames que componen el dataset con los valores a obtener
            en caso de que no corresponda se inserta None
        """
        self.datasets = datasets
        self.pasado = pasado
        self.futuro = futuro
        self.shuffle = shuffle
        self.inicio = min([_.index.min() for _ in datasets])
        self.fin = max([_.index.max() for _ in datasets])
        self.longitud = self.fin -self.inicio + 1
        self.i = 0
        self.vector = list(range(self.pasado, self.longitud - self.futuro, 1)) + self.inicio
      
    def __iter__(self):
        
        if self.shuffle:
            np.random.shuffle(self.vector)
        self.i = 0
        return(self)
    
    def __next__(self) -> None or list:
        if self.i >= len(self.vector):
            raise StopIteration
        else:
            i = self.i
            self.i += 1
        return [self.vector[i] if (self.vector[i] >= df.index.min() + self.pasado) and
                (self.vector[i] <= df.index.max() - self.futuro) else None for df in self.datasets]
            
    def __len__(self) -> int:
        return self.longitud - self.futuro - self.pasado
  

    