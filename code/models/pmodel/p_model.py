import torch
import torch.nn as nn


class RedGeneral(nn.Module):
    """Define una Red Feedforward para la temperatura en una parcela"""

    def __init__(self, Fin: int=None, Fout: int=None, n_layers: int=None, device: str='cpu') -> None:
        """Constructor

        Args:
            Fin (int): Numero de entradas. Defaults to None.
            Fout (int): Numero de salidas. Defaults to None.
            n_layers (int): Numero de capas intermedias. Defaults to None.
            device (str, optional):. Defaults to 'cpu'.
        """
        super().__init__()
        self.n_layers = n_layers
        self.Fin = Fin
        self.Fout = Fout
        self.device = device
        
        assert self.n_layers > 1, "El numero de capas debe ser mayor a 1"
        
        layers = []
        in_features = self.Fin
        for i in range(self.n_layers - 1):
            out_features = in_features
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU())
            in_features = out_features
        layers.append(torch.nn.Linear(in_features, self.Fout))
        self.head = torch.nn.Sequential(*layers)


    def forward(self, input_seq: torch.tensor) -> torch.tensor:
        """Propagacion de la red

        Args:
            input_seq (torch.tensor): Datos de entrada (1, Ly)

        Returns:
            torch.tensor: Datos de salida (1, Ly)
        """
        return self.head(input_seq)