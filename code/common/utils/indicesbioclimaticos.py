#%%
import numpy as np

def mediaMovil(arr: np.ndarray, periodo: int) -> np.ndarray:
    """Calcula la media movil de n periodos

    Args:
        arr (np.ndarray): array
        periodo (int)

    Returns:
        np.ndarray: media movil
    """
    ret = np.cumsum(arr, dtype=float)
    ret[periodo:] = ret[periodo:] - ret[:-periodo]
    ret[periodo-1:] = ret[periodo - 1:] / periodo
    for i in range(1, periodo):
        ret[i-1] = np.sum(arr[0:i]) / i
    return ret

def sumaMovil(arr: np.ndarray, periodo: int) -> np.ndarray:
    """Calcula la suma movil de n periodos

    Args:
        arr (np.ndarray): array
        periodo (int)

    Returns:
        np.ndarray: suma movil
    """
    ret = np.cumsum(arr, dtype=float)
    ret[periodo:] = ret[periodo:] - ret[:-periodo]
    for i in range(1, periodo):
        ret[i-1] = np.sum(arr[0:i])
    return ret

def maximaMovil(arr: np.ndarray, periodo: int) -> np.ndarray:
    """Calcula la maxima movil de n periodos

    Args:
        arr (np.ndarray): array
        periodo (int)

    Returns:
        np.ndarray: suma movil
    """
    return np.array([np.max(arr[i - periodo + 1:i + 1])
                     if i >= periodo else np.max(arr[0: i + 1]) 
                     for i in range(len(arr))])

def biotemperatura(temperatura: np.ndarray, muestras_dia: int=1) -> np.ndarray:
    """Es un concepto creado por Holdridge (1947), y uno de los 
    valores que se tiene en cuenta en su clasificación de las formas 
    de vida, en él que se da mucha importancia a las temperaturas. Sus fórmulas son:
    Biotemperatura 1 = Σ ti/365 
    en donde ti = temperaturas medias diarias que superan los 0ºC y no exceden los 30ºC.

    Args:
        temperatura (np.ndarray): temperaturas diarias
        muestras_dia (int): en caso de la frecuencia no se diaria se especifica la frecuencia

    Returns:
        np.ndarray: biotemperatura
    """
    temperatura[np.where(temperatura > 30)] = 0
    temperatura[np.where(temperatura < 0)] = 0
    return mediaMovil(temperatura, 365 * muestras_dia)

def integral_termica(temperatura: np.ndarray) -> np.ndarray:
    """Integral térmica negativa, la suma de las valores de la temperatura
    dividido por el numero de terminos

    Args:
        temperatura (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    return np.cumsum(temperatura) / np.arange(1, len(temperatura) + 1)

def indice_lasser(precipitacion: np.ndarray, temperatura: np.ndarray, muestras_dia: int=1) -> np.ndarray:
    """
    Índice de Lasser = Pm/Tm
    en donde, Pm = Precipitación mensual y Tm temperatura media mensual

    Args:
        precipitacion (np.ndarray): Precipitación mensual (30 dias anteriores)
        temperatura (np.ndarray): Temperatura media mensual (30 dias anteriores)
        muestras_dia (int): en caso de la frecuencia no se diaria se especifica la frecuencia

    Returns:
        np.ndarray: Tndice de Lasser
    """
    return mediaMovil(precipitacion, 30 * muestras_dia) / mediaMovil(temperatura, 30 * muestras_dia)

def indice_lang(precipitacion: np.ndarray, temperatura: np.ndarray,  muestras_dia: int=1) -> np.ndarray:
    """Índice de Lang = P/T
    en donde P = Precipitaciones anuales (mm) y T = Temperatura media anual (ºC).

    Args:
        precipitacion (np.ndarray): precipitacion media anual (365 dias)
        temperatura (np.ndarray): temperatura media anual (365 dias)
        muestras_dia (int): en caso de la frecuencia no se diaria se especifica la frecuencia

    Returns:
        np.ndarray:  Indice de Lang
    """
    return mediaMovil(precipitacion, 365 * muestras_dia) / mediaMovil(temperatura, 365 * muestras_dia)

def indice_angtrom(precipitacion: np.ndarray, temperatura: np.ndarray, muestras_dia: int=1) -> np.ndarray:
    """Índice de Angström: es igual a la precipitación media mensual
    multiplicada por 1,07, elevado a la temperatura media mensual con signo negativo.
    Índice de Angström = P * 1,07^(-T)
    en donde P = Precipitaciones mensual (mm) y T = Temperatura media mensual (ºC).

    Args:
        precipitacion (np.ndarray): Precipitación mensual (30 dias anteriores)
        temperatura (np.ndarray): Temperatura media mensual (30 dias anteriores)
        muestras_dia (int): en caso de la frecuencia no se diaria se especifica la frecuencia

    Returns:
        np.ndarray: Tndice de Angtrom
    """
    P = mediaMovil(precipitacion, 30 * muestras_dia)
    T = mediaMovil(temperatura, 30 * muestras_dia)
    return P * np.power(1.07, T)

def indice_gasparin(precipitacion: np.ndarray, temperatura: np.ndarray, muestras_dia: int=1) -> np.ndarray:
    """Índice de Gasparín: es igual a la precipitación anual entre la temperatura media anual
    Indice de Gasparín = P / (50·T)
    en donde P es la precipitación anual total (en mm.) y T es la temperatura media anual (en Cº).

    Args:
        precipitacion (np.ndarray): precipitacion media anual (365 dias)
        temperatura (np.ndarray): temperatura media anual (365 dias)
        muestras_dia (int): en caso de la frecuencia no se diaria se especifica la frecuencia

    Returns:
        np.ndarray: Indice de Gasparin
    """
    P = mediaMovil(precipitacion, 365 * muestras_dia)
    T = mediaMovil(temperatura, 365 * muestras_dia)
    return P / (50 * T)

def indice_martonne(precipitacion: np.ndarray, temperatura: np.ndarray, muestras_dia: int=1) -> np.ndarray:
    """Índice de Martonne o índice de aridez de Martonne (1926). Permite una primera i
    dentificación fitoclimática del mundo, aunque es especialmente efectivo en zonas tropicales y subtropicales. Puede calcularse el índice anual o el mensual cuyas fórmulas son:
    IMmensual = [Pm/(Tm+10)]*12
    en donde Pm = Precipitación media mensual en mm; Tm = Temperatura media mensual en grados centígrados.

    Args:
        precipitacion (np.ndarray): Precipitación mensual (30 dias anteriores)
        temperatura (np.ndarray): Temperatura media mensual (30 dias anteriores)
        muestras_dia (int): en caso de la frecuencia no se diaria se especifica la frecuencia

    Returns:
        np.ndarray: Tndice de Angtrom
    """
    P = mediaMovil(precipitacion, 30 * muestras_dia)
    T = mediaMovil(temperatura, 30 * muestras_dia)
    return (P / (10 + T)) * 12

def indice_birot(precipitacion: np.ndarray, temperatura: np.ndarray,  muestras_dia: int=1) -> np.ndarray:
    """Índice de Birot 
    Índice de Birot = (10 – (n*p/t)
    en donde: n = número medio mensual de días con precipitación;
    p = precipitación media mensual;
    t = temperatura media mensual.

    Args:
        precipitacion (np.ndarray): Precipitación mensual (30 dias anteriores)
        temperatura (np.ndarray): Temperatura media mensual (30 dias anteriores)
        muestras_dia (int): en caso de la frecuencia no se diaria se especifica la frecuencia

    Returns:
        np.ndarray: Indice de Birot
    """
    P = mediaMovil(precipitacion, 30 * muestras_dia)
    T = mediaMovil(temperatura, 30 * muestras_dia)
    N = np.where(precipitacion, 0, 1)
    return (10 - (N * P / T))

def indice_hugling(temperatura: np.ndarray,  muestras_dia: int=1) -> np.ndarray:
    """Indice basado en la insolación y temperatura de una zona que se utiliza
    en viticultura para determinar la idoneidad de una zona para el cultivo de la vid.
    Se calcula como:
    IH = (suma) K * [((Ta-10ºC)+(T-10ºC))/2]
    Desde el 1 de Abril al 30 de Septiembre (periodo activo)
    Donde:
    IH: índice de posibilidades heliotérmicas
    Ta: temperatura activa
    T: temperatura máxima diaria
    K: longitud de los días (entre 1,02 y 1,06 equivalentes a 40º y 50º de latitud)

    Para que una zona sea apta para el cultivo de la vid, su ndice de posibilidades
    heliotérmicas debe estar comprendido entre 1.500 y 2.500

    Args:
        temperatura (np.ndarray): Temperatura media mensual (30 dias anteriores)
        muestras_dia (int): en caso de la frecuencia no se diaria se especifica la frecuencia

    Returns:
        np.ndarray: Indice de Hugling
    """
    T = maximaMovil(temperatura, 1 * muestras_dia)
    return 1.03 * ((temperatura -10 ) + (T -10)) / 2
# %%
