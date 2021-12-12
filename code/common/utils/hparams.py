from pathlib import Path
import models.seq2seq.seq2seq_2layers_2heads
import yaml

def save_hyperparameters(model: models.seq2seq.seq2seq_2layers_2heads.EncoderDecoderWrapper=None, path: Path=None):
    """Guarda los hyperparametros del modelo en un archivo

    Args:
        model (models.seq2seq.seq2seq_2layers_2heads.EncoderDecoderWrapper): Modelo. Defaults to None.
        path (Path, optional): Ruta del archivo donde guardar los hyperparametros. Defaults to None.
    """
    dict = {"encoder": {"sequence_len": model.encoder.sequence_len,
                        "hidden_size": model.encorder.hidden_size,
                        "input_feature_len": model.encore.input_feature_len,
                        "rnn_num_layers": model.encoder.num_layers,
                        "rnn_directions": model.encoder.rnn_directions,
                        "rnn_dropout": model.encoder.rnn_dropout
                    },
            "decoder": {"input_feature_len": model.decoder.input_feature_len,
                        "hidden_size": model.decoder.hidden_size,
                        "output_size": model.decoder.output_size,
                        "bin_len": model.decoder.self.bins_len,
                        "dropout": model.decoder.dropout      
                    },
            "model": {"output_size": model.output_size,
                      "output_sequence_len": model.output_sequence_len
                      }
            }
    
    path.mkdir(parents=True, exist_ok=True)  
    with open(path, 'wb') as handler:
        yaml.safe_dump(dict, handler, allow_unicode=True)

def load_hyperparameter(path: Path=None) -> dict:
    """Carga los parametros

    Args:
        path (Path, optional): Ruta del archivo donde se han guardado los hyperparametros
        
    Return:
        dict
    """
    with open(path, 'rb') as handler:
        dictionario = yaml.safe_load(handler)
    return dictionario