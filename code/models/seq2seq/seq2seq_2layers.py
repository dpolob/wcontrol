import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    """Define una RNN basada en GRU"""

    def __init__(self, rnn_num_layers: int, input_feature_len: int, sequence_len: int, hidden_size: int = 100, bidirectional: bool = False, device: str = 'cpu', rnn_dropout: float = 0.0) -> None:
        """Clase para el encoder. Toma como parametros de entrada:
            - Variables de estacion y temporales Xt (N, Lx, Ft in)
            - Variables calculadas X (N, Lx, Ff in)
        La salida es el estado oculto de la ultima capa

        Inputs:
            rnn_num_layers (int): numero de capas de la red (NL)
            input_feature_len (int): numero de features de cada secuencia (Ft in)
            sequence_len (int): longitud de la secuencia temporal (Lx)
            hidden_size (int): numero de representaciones internas de la red (HID)
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

    def forward(self, input_seq: torch.tensor) -> tuple:
        """
        Propagacion de la red

        Inputs:
            input_seq (tensor): (N, Lx, Hin) secuencia con datos de entrada.
                En caso de batch_first=False (Lx, N, Hin)
        Outputs:
            gru_out (tensor): (N, L, HID) correspondiente al estado del ultima capa
                La salida de la red (N, Lx, D * HID) es convertida a (N, Lx, HID) en el codigo
            hidden (tensor): (N, HID) 
                La salida hidden de la red es (D * NL, N, HID) y es convertida (N, HID) en el codigo
        """

        assert input_seq.ndim == 3, "dims(input) no es 3, batch first???!!!"
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0),
                         self.hidden_size, device=self.device, dtype=torch.float)
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

    def __init__(self, input_feature_len: int, hidden_size: int, dropout: float = 0.2) -> None:
        """
        Input:
            input_feature_len (int): Numero de features de entrada (Ft in)
            hidden_size (int): Numero de representaciones internas de la red (HID) 
            dropout (float): valor de dropout para apagar la capa hidden 
        """

        super().__init__()
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=input_feature_len,
            hidden_size=hidden_size,
        )
        self.out1 = nn.Linear(hidden_size, 50)
        self.relu = nn.ReLU()
        self.out = nn.Linear(50, 1)
        self.attention = False
        self.dropout = nn.Dropout(dropout)

    def forward(self, y: torch.tensor, prev_hidden: torch.tensor) -> tuple:
        """
        Propagacion de la red

        Inputs:
            y (tensor): (N, Ft in) Features de datos de entrada 
        Outputs:
            output (tensor): (N, 1) salida de la red
            rnn_hidden (tensor): (N, HID) salida de la representacion de la red 
                La salida hidden de la red es (D * NL, N, HID) y es convertida (N, HID) en el codigo
        """
        rnn_hidden = self.decoder_rnn_cell(y, prev_hidden)  # (N, HID)
        output = self.out(self.relu(self.out1(rnn_hidden)))  # (N, 1)
        return output, self.dropout(rnn_hidden)  # (N,1), (N, HID)


class EncoderDecoderWrapper(nn.Module):
    """Wrapper para el modelo"""

    def __init__(self, encoder: RNNEncoder, decoder_cell: DecoderCell, output_size: int, output_sequence_len: int, teacher_forcing: float = 0.3, duplicate_teaching: int = 40, device: str = 'cpu') -> None:
        """ Definicion del modelo

        Inputs:
            encoder (RNNEncoder): Modelo de encoder a usar
            decoder_cell (DecoderCell): Modelo de decoder a usar
            output_size (int): Numero de variables de salida (Fout)
            output_sequence_len (int): Longitud de la secuencia a predecir (Ly)
            teacher_forcing (float, optional): Probabilidad de uso del teacher. Defaults to 0.3.
            duplicate_teaching (int, optional): Duplicar la probabilidad pasado un determinado punto en la secuencia. Defaults to 40.
            device (str, optional): Dispositivo. Defaults to 'cpu'.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.output_size = output_size
        self.output_sequence_len = output_sequence_len
        self.teacher_forcing = teacher_forcing
        self.duplicate_teaching = duplicate_teaching
        self.device = device

    def forward(self, x_t: torch.tensor, x: torch.tensor, y_t: torch.tensor, y: torch.tensor = None, p: torch.tensor = None, teacher: bool = None) -> torch.tensor:
        """Propagacion

        Inputs:
            x_t (torch.tensor): Features pasadas de estacion y tiempo (N, Lx, Ft in)
            x (torch.tensor): Features pasadas de climaticas y calculadas(N, Lx, Ff in)
            y_t (torch.tensor): Features futuras de estacion y tiempo (N, Ly, Ft in)
            y (torch.tensor): Features futuras climatica (N, Ly, Fout)
            p (torch.tensor): Features climaticas de prediccion (N, Ly, Fout)
            teacher (bool, optional): Define si se usa el teacher. Defaults to None.

        Outputs:
            outputs: Salida de la red(N, Ly, Fout)
        """
        encoder_input = torch.cat((x_t, x), 2)  # (N, Lx, Ft in + Ff in))
        _, encoder_hidden = self.encoder(encoder_input)  # (N, HID)

        decoder_hidden = encoder_hidden  # (N, HID)
        # (N, Ft in) + (N, Fout) = (N,Ft in + Fout)   ### + Fout
        decoder_input = torch.cat((y_t[:, 0, :], y[:, 0, :]), 1)

        # elimino la componente extra que me sobra
        y_t = y_t[:, 1:, :]
        if y is not None:
            y = y[:, 1:, :]
        if p is not None:
            p = p[:, 1:, :]

        outputs = torch.zeros(size=(x_t.size(0), self.output_sequence_len, self.output_size), device=self.device, dtype=torch.float)  # (N, Ly, 1)
        #  tf = self.teacher_forcing 
        for i in range(self.output_sequence_len):
            # (N,1), (N, HID) = decoder((N,Ft in + Fout),(N, HID))
            decoder_output, decoder_hidden = self.decoder_cell(decoder_input, decoder_hidden)
            outputs[:, i, :] = decoder_output
            # if i == self.duplicate_teaching:
            #    tf = self.teacher_forcing * 2
            # if (teacher) and (y is not None) and (i > 0) and (torch.rand(1) < tf):
            #    decoder_input = torch.cat((y_t[:, i, :], y[:, i, :]), axis=1)  # (N,Ft in) + (N,Fout) = (N,Ft in + Fout)
            # else:
            #    decoder_input = torch.cat((y_t[:, i, :], decoder_output), 1)  # (N,Ft in) + (N,Fout) = (N,Ft in + Fout)
            
            # asignacion de P
            decoder_input = torch.cat((y_t[:, i, :], p[:, i, :]), axis=1)  # (N,Ft in) + (N, Fout) = (N,Ft in + Fout)
        return outputs  # (N, Ly, Fouts)
