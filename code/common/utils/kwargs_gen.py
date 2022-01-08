from attrdict import AttrDict
from datetime import datetime

class generar_kwargs():
    
    def _dataloader(self, model: str=None, fase: str=None, cfg: AttrDict=None, **kwargs) -> dict:
        """Genera los argumentos para las diferentes funciones, evitando repetir codigo

        Args:
            model (str, optional): 'zmodel' o 'pmodel'
            dataloader (bool, optional): Si true genera argumentos para un dataloader
            fase (str, optional): 'validation' para dataset de validacion, 'train' para dataset de train. Defaults to None.
            cfg (AttrDict, optional): archivo de configuracion
            **kwargs (dict): parametros opcionales necesarios

        Returns:
            dict: diccionario kwargs 
        """
        metadata = kwargs.get('metadata', None)
        datasets = kwargs.get('datasets', None)
        
        if model not in ['zmodel', 'pmodel'] or fase not in ['validation', 'train', 'test'] or cfg is None or metadata is None or datasets is None:
            print("la funcion generar_kwargs no tiene los parametros correctos")
            exit()
        
        if model == 'pmodel' and fase == 'train':
            handler = cfg.pmodel.dataloaders.train
        elif model == 'pmodel' and fase == 'validation':
            handler = cfg.pmodel.dataloaders.validation
        elif model == 'pmodel' and fase == 'test':
            handler = cfg.pmodel.dataloaders.test
        elif model == 'zmodel' and fase == 'train':
            handler = cfg.zmodel.dataloaders.train
        elif model == 'zmodel' and fase == 'validation':
            handler = cfg.zmodel.dataloaders.validation
        elif model == 'zmodel' and fase == 'test':
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
