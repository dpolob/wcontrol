#%%


import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
    
 
class PModelDataset(Dataset):
    """
        
    """
    def __init__(self, datasets: list, componentes: slice) -> tuple:
        """Dataset especifico para pmodel

        Args:
            datasets (list): [Predicciones Zmodel (len, Ly, Fout), Valores Reales (len, Ly, Fout)]
            componentes (slice): Componente a ofrecer dentro de Fout, normalmente 0=temperatura, 1=hr y [2-10]=clases de lluvia

        Returns:
            tuple: (len(idx), Ly, len(componentes), (len(idx), Ly, len(componentes))
        """
        assert isinstance(datasets, list), "datasets no es una lista"
        assert isinstance(componentes, slice), "componentes no es un slice"
        assert len(datasets) == 2, "Los componentes del dataset no son 2"
        assert datasets[0].shape == datasets[1].shape, "Los shapes de cada componente es diferente"
        self.X = datasets[0]
        self.Y = datasets[1]
        self.componentes = componentes

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: list) -> torch.tensor:
        
        X = self.X[idx, :, self.componentes]
        Y = self.Y[idx, :, self.componentes]
        return (torch.from_numpy(X).float(),  # (72, Len(componentes))
                torch.from_numpy(Y).float())  # (72, Len(componentes))

