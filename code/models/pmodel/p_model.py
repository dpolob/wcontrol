import torch
import torch.nn as nn


class RedGeneral(nn.Module):
    
    def __init__(self, Fin: int, Fout: int, hidden_layers: dict, hidden_size: int, gru_num_layers: int) -> None:
        """GRU + FeedForward

        Inputs:
            Fin (int): numero de entradas
            Fout (int): numero de salidas
            hidden_size (int): numero de representaciones internas de la red
            gru_num_layers (int): capas de la red GRU
            hidden_size (int): Numero de representaciones internas de la red (HID) 
            hidden_layers (dict): Capas ocultas de la head de regresion.
                {0: numero de neuronas capa 0,
                 1: numero de neuronas capa 1,
                 ...}
        """
        super().__init__()
        self.gru = nn.GRU(Fin, hidden_size=hidden_size, num_layers=gru_num_layers, batch_first = True)
        layers = []
        in_features = hidden_size
        for i in range(len(hidden_layers)):
            out_features = int(hidden_layers[i])
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU())
            in_features = out_features
        layers.append(torch.nn.Linear(in_features, Fout))
        self.head = torch.nn.Sequential(*layers)
            
    def forward(self, input_seq):
        gru_out, _ = self.gru(input_seq)
        out  = self.head(gru_out)
        return out
