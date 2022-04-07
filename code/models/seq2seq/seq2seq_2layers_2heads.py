import torch
import torch.nn as nn
from common.utils.smelu import SmeLU

class RNNEncoder(nn.Module):
    """Define una RNN basada en GRU"""

    def __init__(self, rnn_num_layers: int, input_feature_len: int, sequence_len: int, hidden_size: int,
                 rnn_dropout: float, bidirectional: bool = False, device: str = 'cpu') -> None:
        """Clase para el encoder. Toma como parametros de entrada:
           La salida es el estado oculto de la ultima capa

        Inputs:
            rnn_num_layers (int): numero de capas de la red GRU (NL)
            input_feature_len (int): numero de features de cada secuencia (Ff)
            sequence_len (int): longitud de la secuencia temporal del pasado (Lx)
            hidden_size (int): numero de representaciones internas de la red (HID)
            bidirectional (bool): define si la red es bidireccional (D)
            device (str): 'cpu' o 'cuda' para seleccionar dispositivo de calculo
            rnn_dropout (float): valor de dropout, solo valido si rnn_num_layers > 1
        """
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
        if self.device == 'cuda' and not torch.cuda.is_available():  # esto hace que modelos entrenados en GPU corran en CPUS
            self.device = 'cpu'  
            
        assert input_seq.ndim == 3, f"{self.__class__}. dims(input) no es 3, batch first???!!!"
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
            if self.num_layers > 0:
                hidden = hidden[-1]
            else:
                hidden = hidden.squeeze(0)
            hidden = hidden.sum(axis=0)
        else:
            hidden.squeeze_(0)
        return gru_out, hidden

class DecoderCell(nn.Module):
    """Define una celda GRU"""

    def __init__(self, input_feature_len: int, hidden_size: int, output_size: int, bins_len: int, dropout: float,
                  regres_hidden_layers: dict,  class_hidden_layers: dict) -> None:
        """
        Input:
            input_feature_len (int): Numero de features de entrada (Ft + Fnwp) Fnwp es la prediccion, en este caso (nwp_temp, nwp_hr, nwp_clase_precipitacion[0-7])
            hidden_size (int): Numero de representaciones internas de la red (HID) 
            output_size (int): Numero de variables de salida (Fout), en este caso (nwp_temp, nwp_hr, nwp_clase_precipitacion[0-7])
            bins_len (int): Numero de clases para la clasificacion
            regres_hidden_layers (dict): Capas ocultas de la head de regresion.
                {0: numero de neuronas capa 0,
                 1: numero de neuronas capa 1,
                 ...}
            class_hidden_layers (dict): Capas ocultas de la head de clasificacion.
                {0: numero de neuronas capa 0,
                 1: numero de neuronas capa 1,
                 ...}
            dropout (float): valor de dropout para apagar la capa hidden 
        """
        super().__init__()
        self.input_feature_len = int(input_feature_len)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.bins_len = int(bins_len)
        self.dropout = dropout      
        self.regres_hidden_layers = len(regres_hidden_layers)
        self.class_hidden_layers =len(class_hidden_layers)
        self.decoder_rnn_cell = nn.GRUCell(input_size=self.input_feature_len, hidden_size=self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Head para temperatura y hr (Regresion)
        temphr = []
        in_features = self.hidden_size
        for i in range(self.regres_hidden_layers):
            out_features = int(regres_hidden_layers[i])
            temphr.append(torch.nn.Linear(in_features, out_features))
            temphr.append(SmeLU(beta=1.1))
            in_features = out_features
        temphr.append(torch.nn.Linear(in_features, output_size - self.bins_len))
        self.temphr_head = torch.nn.Sequential(*temphr)

        # # Head para precipitacion (Clasificacion)
        class_head = []
        in_features = self.hidden_size
        for i in range(self.class_hidden_layers):
            out_features = int(class_hidden_layers[i])
            class_head.append(torch.nn.Linear(in_features, out_features))
            class_head.append(SmeLU(beta=1.1))
            in_features = out_features
        class_head.append(torch.nn.Linear(in_features, self.bins_len))
        self.class_head = torch.nn.Sequential(*class_head)

    def forward(self, y: torch.tensor, prev_hidden: torch.tensor) -> tuple:
        """
        Propagacion de la red

        Inputs:
            y (tensor): (N, Ft + Fnwp) Features de datos de entrada, compuesto de temporales + las predicciones nwp
        Outputs:
            output (tensor): (N, Fout + 1 ) son 2 (temp + hr) + (class_rain)
            rnn_hidden (tensor): (N, HID) salida de la representacion de la red 
                La salida hidden de la red es (D * NL, N, HID) y es convertida (N, HID) en el codigo
        """
       
        rnn_hidden = self.decoder_rnn_cell(y, prev_hidden)  # (N, HID)
        temphr_output = self.temphr_head(rnn_hidden)  # (N, 2)
        class_rain_output = self.class_head(rnn_hidden)  # (N, 1)
        
        output = torch.cat([temphr_output, class_rain_output], dim=-1)  # (N, Fout)
        return output, self.dropout(rnn_hidden)  # (N,1), (N, HID)

class EncoderDecoderWrapper(nn.Module):
    """Wrapper para el modelo"""

    def __init__(self, encoder: RNNEncoder, decoder_cell: DecoderCell, output_size: int, output_sequence_len: int, device: str = 'cpu') -> None:
        """ Definicion del modelo

        Inputs:
            encoder (RNNEncoder): Modelo de encoder a usar
            decoder_cell (DecoderCell): Modelo de decoder a usar
            output_size (int): Numero de variables de salida (Fout)
            output_sequence_len (int): Longitud de la secuencia a predecir (Ly)
            device (str, optional): Dispositivo. Defaults to 'cpu'.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.output_size = output_size
        self.output_sequence_len = output_sequence_len
        self.device = device

    def forward(self, x_f: torch.tensor, x: torch.tensor, y_t: torch.tensor, y: torch.tensor = None, p: torch.tensor = None) -> torch.tensor:
        """Propagacion

        Inputs:
            x_f (torch.tensor): Features de entrada (N, Lx, Ff)  Ff incluye las calculadas, climaticas y las temporales(Ft)
            x (torch.tensor): Variables climaticas a predecir  (N, Lx, Fout )
            y_t (torch.tensor): Features futuras de estacion y tiempo (N, Ly, Ft)
            y (torch.tensor): Features futuras climatica (N, Ly, Fout)
            p (torch.tensor): Features climaticas de prediccion (N, Ly, Fout)

        Outputs:
            outputs: Salida de la red(N, Ly, Fout)
        """
        if self.device == 'cuda' and not torch.cuda.is_available():  # esto hace que modelos entrenados en GPU corran en CPUS
            self.device = 'cpu'  
        
        encoder_input = x_f  #  (N, Lx, Ff)
        _, encoder_hidden = self.encoder(encoder_input)  # (N, HID)

        decoder_hidden = encoder_hidden  # (N, HID)
        
        
 
         # elimino la componente extra que me sobra
        y_t = y_t[:, :-1, :]
        if y is not None:
            y = y[:, :-1, :]
        if p is not None:
            p = p[:, :-1, :]
            
        # (N, Ft + Fout) = (N, Ft) + (N, Fout) 
        decoder_input = torch.cat((y_t[:, 0, :], p[:, 0, :]), axis=1)

        outputs = torch.zeros(size=(x_f.size(0), self.output_sequence_len, self.output_size), device=self.device, dtype=torch.float)  # (N, Ly, Fout (hr + temp + 8 clases)
        for i in range(self.output_sequence_len):
            # (N,1), (N, HID) = decoder((N,Ft + Fout), (N, HID))
            decoder_output, decoder_hidden = self.decoder_cell(decoder_input, decoder_hidden)
            outputs[:, i, :] = decoder_output  # la salida siempre es decoder_output, no es P
            decoder_input = torch.cat((y_t[:, i, :], p[:, i, :]), axis=1)  # (N,Ft) + (N, Fout) = (N,Ft + Fout)
        return outputs  # (N, Ly, Fout)