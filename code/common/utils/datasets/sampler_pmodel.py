from click.core import batch
import numpy as np
#import pandas as pd
import torch

from torch.utils.data.sampler import Sampler
#from torch.utils.data import Dataset


class PModelSampler(Sampler):
    """Implementa un sampler específico para pmodel
    """
    def __init__(self, datasets: list, batch_size: int=1, shuffle: bool=False) -> list:
        """Devuelve un array con los indices de los muestras a obtener

        Args:
            datasets (list): [Predicciones Zmodel (len, Ly, Fout), Valores Reales (len, Ly, Fout)]
            shuffle (bool, optional): Duevuelve desordenadas las muestras. Defaults to False.

        Returns:
            list: Lista de indices
        """
        assert isinstance(datasets, list), "datasets no es una lista"
        assert len(datasets) == 2, "Las componentes del dataset no son 2"
        assert datasets[0].shape == datasets[1].shape, "Los shapes de cada componente es diferente"
        self.X = datasets[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.vector = list(range(len(self.X)))
      
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
            self.i += self.batch_size
        return self.vector[i : self.i]
            
    def __len__(self) -> int:
        return len(self.X)
  

    