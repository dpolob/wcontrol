from typing import Any


class parser():
    """Clase para los parsers del archivo .yml"""

    def __init__(self, experiment: str=None, epoch: Any=None, prediccion:str=None) -> None:
 
        self.experiment = experiment
        self.epoch = epoch
        self.prediccion = prediccion
    
    def __parser_experiment(self, cfg: dict) -> dict:
        """Parsea la cadena {{experiment}} por name"""

        for key in cfg.keys():
            if isinstance(cfg[key], dict):
                cfg[key] = self.__parser_experiment(cfg[key])
            if isinstance(cfg[key], str):
                cfg[key] = cfg[key].replace("{{experiment}}", self.experiment)
        return cfg

    def __parser_epoch(self, cfg: dict) -> dict:
        """Parsea la cadena {{epoch}} de los archivos de configuracion yml"""
        
        if isinstance(self.epoch, int):
            self.epoch = str(self.epoch)  # epoch es un numero
        for key in cfg.keys():
            if isinstance(cfg[key], dict):
                cfg[key] = self.__parser_epoch(cfg[key])
            if isinstance(cfg[key], str):
                 cfg[key] =  cfg[key].replace("{{epoch}}",  self.epoch)
        return  cfg
    
    def __parser_prediccion(self, cfg: dict) -> dict:
        """Parsea la cadena {{prediccion}} de los archivos de configuracion yml"""
        for key in cfg.keys():
            if isinstance(cfg[key], dict):
                cfg[key] = self.__parser_prediccion(cfg[key])
            if isinstance(cfg[key], str):
                cfg[key] =  cfg[key].replace("{{prediccion}}",  self.prediccion)
        return cfg

    def __call__(self, cfg: dict) -> dict:
        if self.experiment is not None:
            cfg = self.__parser_experiment(cfg)
        if self.epoch is not None:
            cfg = self.__parser_epoch(cfg)
        if self.prediccion is not None:
            cfg = self.__parser_prediccion(cfg)
        return cfg