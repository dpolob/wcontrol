from attrdict import AttrDict
from datetime import datetime

class generar_kwargs():
    
    def _dataloader(self, modelo: str, fase: str, cfg: AttrDict, **kwargs) -> dict:
        """Genera los argumentos para las diferentes funciones, evitando repetir codigo

        Args:
            modelo (str): 'zmodel' o 'pmodel'
            fase (str): 'validation' para dataset de validacion, 'train' para dataset de train. 'test' para dataset
                de test
            cfg (AttrDict): archivo de configuracion
            **kwargs (dict): parametros opcionales necesarios

        Returns:
            dict: diccionario kwargs 
        """
        metadata = kwargs.get('metadata', None)
        datasets = kwargs.get('datasets', None)
        
        if modelo not in ['zmodel', 'pmodel'] or fase not in ['validation', 'train', 'test'] or cfg is None or metadata is None or datasets is None:
            print("La funcion generar_kwargs._predict() no tiene los parametros correctos")
            exit()
        
        if modelo == 'pmodel' and fase == 'train':
            handler = cfg.pmodel.dataloaders.train
        elif modelo == 'pmodel' and fase == 'validation':
            handler = cfg.pmodel.dataloaders.validation
        elif modelo == 'pmodel' and fase == 'test':
            handler = cfg.pmodel.dataloaders.test
        elif modelo == 'zmodel' and fase == 'train':
            handler = cfg.zmodel.dataloaders.train
        elif modelo == 'zmodel' and fase == 'validation':
            handler = cfg.zmodel.dataloaders.validation
        elif modelo == 'zmodel' and fase == 'test':
            handler = cfg.zmodel.dataloaders.test
        else:
            raise NotImplementedError
        
        data = {'datasets': datasets, 'fecha_inicio_test': datetime.strptime(handler.fecha_inicio, "%Y-%m-%d %H:%M:%S"), 
                'fecha_fin_test': datetime.strptime(handler.fecha_fin, "%Y-%m-%d %H:%M:%S"), 
                'fecha_inicio': datetime.strptime(metadata['fecha_min'], "%Y-%m-%d %H:%M:%S"),
                'pasado': cfg.pasado,
                'futuro': cfg.futuro,
                'etiquetaX': list(cfg.prediccion), 'etiquetaF': list(cfg.zmodel.model.encoder.features),
                'etiquetaT': list(cfg.zmodel.model.decoder.features), 'etiquetaP': list(cfg.zmodel.model.decoder.nwp),
                'indice_max': metadata['indice_max'], 'indice_min': metadata['indice_min']
                }  
        
        return data
    
    def _predict(self, modelo: str, cfg: AttrDict) -> dict:
        """Genera los argumentos para las diferentes funciones de prediccion, evitando repetir codigo

        Args:
            modelo (str, optional): 'zmodel' o 'pmodel'
            cfg (AttrDict, optional): archivo de configuracion
        
        Returns:
            dict: diccionario kwargs 
        """
        if modelo == 'zmodel':
            data = {'device': 'cuda' if cfg.zmodel.model.use_cuda else 'cpu',
                    'path_checkpoints': cfg.paths.zmodel.checkpoints,
                    'use_checkpoint': cfg.zmodel.dataloaders.test.use_checkpoint,
                    'path_model' : cfg.paths.zmodel.model
                    }
        elif modelo == 'pmodel':
            data = {'device': 'cuda' if cfg.pmodel.model.use_cuda else 'cpu',
                    'path_checkpoints': cfg.paths.pmodel.checkpoints,
                    'use_checkpoint': cfg.pmodel.dataloaders.test.use_checkpoint,
                    'path_model' : cfg.paths.pmodel.model
                    }
        else:
            raise NotImplementedError
        return data 
