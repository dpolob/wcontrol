import pandas as pd
import numpy as np

def macd(arr: np.ndarray, corta: int=12, larga: int=26, señal: int=9) -> tuple:
    """Calcula el macd de una secuenca

    Args:
        arr (np.ndarray): secuencia de datos
        corta (int, optional): Periodo corto de macd. Defaults to 12.
        larga (int, optional): Periodo largo de macd. Defaults to 26.
        señal (int, optional): Periodo de la seña. Defaults to 9.

    Returns:
        tuple: (np.ndarray, np.ndarray): (macd, macd-señal)
    """
    df = pd.Series(arr)
    exp1 = df.ewm(span=corta, adjust=False).mean()
    exp2 = df.ewm(span=larga, adjust=False).mean()
    macd = exp1-exp2
    exp3 = macd.ewm(span=señal, adjust=False).mean()
    macd_señal = macd - exp3
    
    return np.array(macd.values) - np.array(macd_señal.values)