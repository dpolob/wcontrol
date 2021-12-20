import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, rnn_num_layers: int, input_feature_len: int, sequence_len: int, hidden_size: int = 100, bidirectional: bool = False, device: str = 'cpu', rnn_dropout: float = 0.0) -> None:
        super().__init__()
        self.sequence_len = int(sequence_len)
        self.hidden_size = int(hidden_size)
        self.input_feature_len = int(input_feature_len)
        self.rnn_num_layers = int(rnn_num_layers)
        self.bidirectional = bidirectional
        self.rnn_directions = 2 if self.bidirectional else 1
        self.rnn_dropout = rnn_dropout
      
        self.gru = nn.GRU(
            num_layers=self.rnn_num_layers,
            input_size=self.input_feature_len,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.rnn_dropout if self.rnn_num_layers>1 else 0.0
        )
        self.device = device

    def forward(self, input_seq: torch.tensor) -> tuple:
        assert input_seq.ndim == 3, "dims(input) no es 3, batch first???!!!"
        ht = torch.zeros(self.rnn_num_layers * self.rnn_directions, input_seq.size(0),
                         self.hidden_size, device=self.device, dtype=torch.float)
        gru_out, hidden = self.gru(input_seq, ht)
        # codigo para la conversion de las salidas
        if self.rnn_directions * self.rnn_num_layers > 1:
            num_layers = self.rnn_directions * self.rnn_num_layers
            if self.rnn_directions > 1:
                gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
                gru_out = torch.sum(gru_out, axis=2)
            hidden = hidden.view(self.rnn_num_layers, self.rnn_directions, input_seq.size(0), self.hidden_size)
            if self.rnn_num_layers > 0:
                hidden = hidden[-1]
            else:
                hidden = hidden.squeeze(0)
            hidden = hidden.sum(axis=0)
        else:
            hidden.squeeze_(0)
        return gru_out, hidden

class DecoderCell(nn.Module):
    def __init__(self, input_feature_len: int, hidden_size: int, output_size: int=None, bins_len: int=8, dropout: float = 0.2,
                  decoder_temphr_n_layers: int=None,  decoder_class_n_layers: int=None, trial=None ) -> None:
        super().__init__()
        #SINGLE HEAD for temp and hr
        self.input_feature_len = int(input_feature_len)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.bins_len = int(bins_len)
        self.dropout = dropout      
        self.decoder_temphr_n_layers = int(decoder_temphr_n_layers)
        self.decoder_class_n_layers =int(decoder_class_n_layers)
        
        self.decoder_rnn_cell = nn.GRUCell(input_size=self.input_feature_len, hidden_size=self.hidden_size)
        
        temphr = []
        in_features = self.hidden_size
        for i in range(self.decoder_temphr_n_layers):
            out_features = trial.suggest_int(f'decoder_hrtemp_n_units_l{i}', 50 , 250)
            temphr.append(torch.nn.Linear(in_features, out_features))
            temphr.append(torch.nn.ReLU())
            in_features = out_features
        
        temphr.append(torch.nn.Linear(in_features, output_size - self.bins_len))
        self.temphr_head = torch.nn.Sequential(*temphr)

        # 2HEAD for rain
        class_head = []
        in_features = self.hidden_size
        for i in range(self.decoder_class_n_layers):
            out_features = trial.suggest_int(f'decoder_class_n_units_l{i}', 50 , 250)
            class_head.append(torch.nn.Linear(in_features, out_features))
            class_head.append(torch.nn.ReLU())
            in_features = out_features
        
        class_head.append(torch.nn.Linear(in_features, self.bins_len))
        self.class_head = torch.nn.Sequential(*class_head)
              
        #DROPOUT for hidden
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, y: torch.tensor, prev_hidden: torch.tensor) -> tuple:
        rnn_hidden = self.decoder_rnn_cell(y, prev_hidden)  # (N, HID)
        temphr_output = self.temphr_head(rnn_hidden)  # (N, 2)
        class_rain_output = self.class_head(rnn_hidden)  # (N, 1)
        
        output = torch.cat([temphr_output, class_rain_output], dim=-1)  # (N, Fout)
        return output, self.dropout(rnn_hidden)  # (N,1), (N, HID)


class EncoderDecoderWrapper(nn.Module):
    def __init__(self, encoder: RNNEncoder, decoder_cell: DecoderCell, output_size: int, output_sequence_len: int, device: str = 'cpu') -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.output_size = output_size
        self.output_sequence_len = output_sequence_len
        self.device = device

    def forward(self, x_f: torch.tensor, x: torch.tensor, y_t: torch.tensor, y: torch.tensor = None, p: torch.tensor = None) -> torch.tensor:
        encoder_input = x_f  #  (N, Lx, Ff)
        _, encoder_hidden = self.encoder(encoder_input)  # (N, HID)

        decoder_hidden = encoder_hidden  # (N, HID)
        # (N, Ft + Fout) = (N, Ft) + (N, Fout) 
        decoder_input = torch.cat((y_t[:, 0, :], y[:, 0, :]), axis=1)
 
         # elimino la componente extra que me sobra
        y_t = y_t[:, :-1, :]
        if y is not None:
            y = y[:, :-1, :]
        if p is not None:
            p = p[:, :-1, :]

        outputs = torch.zeros(size=(x_f.size(0), self.output_sequence_len, self.output_size), device=self.device, dtype=torch.float)  # (N, Ly, Fout (hr + temp + 8 clases)
        for i in range(self.output_sequence_len):
            # (N,1), (N, HID) = decoder((N,Ft + Fout), (N, HID))
            decoder_output, decoder_hidden = self.decoder_cell(decoder_input, decoder_hidden)
            outputs[:, i, :] = decoder_output  # la salida siempre es decoder_output, no es P
            decoder_input = torch.cat((y_t[:, i, :], p[:, i, :]), axis=1)  # (N,Ft) + (N, Fout) = (N,Ft + Fout)
        return outputs  # (N, Ly, Fout)