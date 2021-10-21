from typing import Any


def parser_experiment(cfg: dict, name: str) -> dict:
    """Parsea la cadena {{experiment}} de los archivos de configuracion yml"""

    for key in cfg.keys():
        if isinstance(cfg[key], dict):
            cfg[key] = parser_experiment(cfg[key], name)
        if isinstance(cfg[key], str):
            cfg[key] = cfg[key].replace("{{experiment}}", name)
    return cfg

def parser_epoch(cfg: dict, epoch: Any=None) -> dict:
    """Parsea la cadena {{epoch}} de los archivos de configuracion yml"""
    if isinstance(epoch, int):
        epoch = str(epoch)  # epoch es un numero
    for key in cfg.keys():
        if isinstance(cfg[key], dict):
            cfg[key] = parser_epoch(cfg[key], epoch)
        if isinstance(cfg[key], str):
            cfg[key] = cfg[key].replace("{{epoch}}", epoch)
    return cfg