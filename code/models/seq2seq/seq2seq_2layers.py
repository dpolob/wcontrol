"""
Modelo Seq2Seq para predicción de series temporales

Clases:
    RNNEncoder
    DecoderCell
    EncoderDecoderWrapper
"""
import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    """Define una RNN basada en GRU"""

    def __init__(self, rnn_num_layers, input_feature_len, sequence_len, hidden_size=100, bidirectional=False, device='cpu', rnn_dropout=0.0):
        """
        Constructor de la RNN
        ---------------------
        Inputs:
            rnn_num_layers (int): numero de capas de la red (NL)
            input_feature_len (int): numero de features de cada secuencia (Hin)
            sequence_len (int): longitud de la secuencia temporal (L)
            hidden_size (int): numero de capas internas de la red (HID)
            bidirectional (bool): define si la red es bidireccional (D)
            device (str): 'cpu' o 'cuda' para seleccionar dispositivo de calculo
            rnn_dropout (float): valor de dropout, solo valido si rnn_num_layers > 1
        """
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout
        )
        self.device = device

    def forward(self, input_seq):
        """
        Propagacion de la red
        ---------------------
        Inputs:
            input_seq (tensor): (N, L, Hin) secuencia con datos de entrada.
                En caso de batch_first=False (L, N, Hin)
        Outputs:
            gru_out (tensor): (N, L, HID) correspondiente al estado del ultima capa
                La salida de la red (N, L, D * HID) es convertida a (N, L, HID) en el codigo
            hidden (tensor): (N, HID) 
                La salida hidden de la res es (D, N, HID) y es convertida (N, HID) en el codigo
        """
        
        assert input_seq.ndim == 3, "dims(input) no es 3, batch first!!!"
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device=self.device, dtype=torch.float)
        gru_out, hidden = self.gru(input_seq, ht)
        # codigo para la conversion de las salidas
        if self.rnn_directions * self.num_layers > 1:
            num_layers = self.rnn_directions * self.num_layers
            if self.rnn_directions > 1:
                gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
                gru_out = torch.sum(gru_out, axis=2)
            hidden = hidden.view(self.num_layers, self.rnn_directions, input_seq.size(0), self.hidden_size)
            if self.num_layers > 0:
                hidden = hidden[-1]
            else:
                hidden = hidden.squeeze(0)
            hidden = hidden.sum(axis=0)
        else:
            hidden.squeeze_(0)

        return gru_out, hidden

# Decoder
class DecoderCell(nn.Module):
    """Define una celda GRU"""
    
    def __init__(self, input_feature_len, hidden_size, dropout=0.2):
        """
        Input:
        input_feature_len
        hidden_size
        dropout=0.2
    """
    
        super().__init__()
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=input_feature_len,
            hidden_size=hidden_size,
        )
        self.out1 = nn.Linear(hidden_size, 50)
        self.out = nn.Linear(50, 1)
        self.attention = False
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, prev_hidden):
        rnn_hidden = self.decoder_rnn_cell(y, prev_hidden) # (N, Hout)
        output = self.out(self.out1(rnn_hidden))  # (N, 1)
        return output, self.dropout(rnn_hidden)  # (N,1), (N, Hout)

# Encoder-Decoder
class EncoderDecoderWrapper(nn.Module):
    """
    Input:
        encoder
        decoder_cell
        output_size

    """
    def __init__(self, encoder: RNNEncoder, decoder_cell: DecoderCell, output_size: int, output_sequence_len: int, teacher_forcing=0.3, duplicate_teaching=40, device='cpu'):
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.output_size = output_size
        self.output_sequence_len = output_sequence_len
        self.teacher_forcing = teacher_forcing
        self.duplicate_teaching = duplicate_teaching
        self.device = device

    def forward(self, x_t, x, y_t, y=None, teacher = None):
        encoder_input = torch.cat((x_t, x), 2)  # (N, L, (x_t+x))
        
        _, encoder_hidden = self.encoder(encoder_input)  # (N, Hout​)
        #print("encoder_hidden:", encoder_hidden.shape)
    
        decoder_hidden = encoder_hidden  # (N, Hout)
        decoder_input = torch.cat((y_t[:,0,:], y[:,0,:]), 1)  # (N, T) + (N, 1) = (N,T+1)
        # elimino la componente extra que me sobra
        y_t = y_t[:,1:, :]
        if y is not None:
            y=y[:, 1:, :]
        #print("decoder_input:", decoder_input.shape)
        
        outputs = torch.zeros(size=(x_t.size(0), self.output_sequence_len, self.output_size), device=self.device, dtype=torch.float)  # (N, L, 1)
        
        tf=self.teacher_forcing
        for i in range(self.output_sequence_len):
            decoder_output, decoder_hidden = self.decoder_cell(decoder_input, decoder_hidden)  # (N,1), (N, Hout) = decoder((N,T+1),(N, Hout))
                
            outputs[:,i, :] = decoder_output

            if i==self.duplicate_teaching:
               tf = self.teacher_forcing * 2
                
            if (teacher) and (y is not None) and (i > 0) and (torch.rand(1) < tf):
                decoder_input = torch.cat((y_t[:, i, :], y[:,i,:]), axis=1)  # (N,T) + (N,1) = (N,T+1)
            else:
                decoder_input = torch.cat((y_t[:, i, :], decoder_output), 1)  # (N,T) + (N,1) = (N,T+1)

        return outputs
