import torch
import numpy as np
from torch.utils.data import Dataset
   
 
class PipelineDataset(Dataset):
    "Data set especifico para Pipeline"
    def __init__(self, datasets: np.ndarray, componentes: slice) -> torch.Tensor:
        """Dataset especifico para pipelines

        Args:
            datasets (np.ndarray): (len(dataset), Ly(futuro), Fout)
            componentes (slice): Componente a ofrecer dentro de Fout, normalmente 0=temperatura, 1=hr y [2-10]=clases de lluvia

        Returns:
            torch.Tensor: ( len(idx), Ly, len(componentes) )
        """
        assert isinstance(datasets, np.ndarray), "datasets no es una array de numpy"
        assert isinstance(componentes, slice), "componentes no es un slice"
        self.X = datasets
        self.componentes = componentes

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: list) -> torch.Tensor:
        X = self.X[idx, :, self.componentes]
        return torch.from_numpy(X).float()  # (Ly, Len(componentes)))
