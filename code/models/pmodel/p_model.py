from unicodedata import bidirectional
import torch
import torch.nn as nn


class RedGeneral(nn.Module):
    # """Define una Red Feedforward para la temperatura en una parcela"""

    def __init__(self, Fin: int=None, Fout: int=None, n_layers: int=None, hidden_size: int=100, gru_num_layers: int=2) -> None:
    
        super().__init__()
        self.gru = nn.GRU(Fin, hidden_size=hidden_size, num_layers=gru_num_layers, batch_first = True)
        layers = []
        in_features = hidden_size
        for i in range(n_layers):
            out_features = in_features
            layers.append(torch.nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU())
            in_features = out_features
        layers.append(torch.nn.Linear(in_features, Fout))
        self.head = torch.nn.Sequential(*layers)
            
    def forward(self, input_seq):
        gru_out, _ = self.gru(input_seq)
        out  = self.head(gru_out)
        return out
