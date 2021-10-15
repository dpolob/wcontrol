def parser_experiment(cfg: dict, name: str) -> dict:
    """Parsea la cadena {{experiment}} de los archivos de configuracion yml

    Args:
        cfg (dict): diccionario
        name (str): reemplazo

    Returns:
        dict: diccionario parseado
    """
    for key in cfg.keys():
        if isinstance(cfg[key], dict):
            cfg[key] = parser_experiment(cfg[key], name)
        if isinstance(cfg[key], str):
            cfg[key] = cfg[key].replace("{{experiment}}", name)
    return cfg