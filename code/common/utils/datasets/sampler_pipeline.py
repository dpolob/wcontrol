import numpy as np
from torch.utils.data.sampler import Sampler


class PipelineSampler(Sampler):
    """Implementa un sampler específico para el pipeline"""
    def __init__(self, datasets: np.ndarray, batch_size: int=1, shuffle: bool=False) -> list:
        """ 
        Args:
            datasets (np.ndarray): (len(dataset), Ly(futuro), Fout)
            batch_size (int): numero de muestras por batch. Defaults to 1.
            shuffle (bool, optional): Duevuelve desordenadas las muestras. Defaults to False.

        Returns:
            list: Lista de indices
        """

        assert isinstance(datasets, np.ndarray), "datasets no es una array de numpy"
        self.X = datasets
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
  

    