import numpy as np


class Escalador():
    """
    Transform features by scaling each feature to a given range.
    This estimator scales and translates each feature individually such that it is in the given range 
    on the training set, e.g. between zero and one.
    where min, max = feature_range.
    This transformation is often used as an alternative to zero mean, unit variance scaling.
    """
    def __init__(self, Xmin: float=None, Xmax: float=None, min: float=0, max: float=1, auto_scale: bool=False):
        """[summary]

        Args:
            Xmin (float): Valor de X minimo, si auto_scale=True Xmin = min(X)
            Xmax (float): Valor de X maximi, si auto_scale=True Xmax = max(X)
            min (float, optional): Valor minimo del escalado. Defaults to 0.
            max (float, optional): Valor máximo del escalado. Defaults to 1.
            auto_scale (bool, optional): Si se toman los valores maximos o minimos de X. Defaults to False.
        """
        if not auto_scale:
            self.Xmin = Xmin
            self.Xmax = Xmax
            self.auto_scale = auto_scale
        else:
            self.auto_scale = auto_scale
        self.min = min
        self.max = max
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Escala X según el patron MaxMin
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

        Args:
            X (np.ndarray): Array a convertir

        Returns:
            [np.ndarray]: Array transformado
        """
        if self.auto_scale:
            self.Xmin = X.min()
            self.Xmax = X.max()
            
        X_std = (X - self.Xmin) / (self.Xmax - self.Xmin)
        X = X_std * (self.max - self.min) + self.min
        return X
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Desescala X según el patron MaxMin
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

        Args:
            X (np.ndarray): Array a convertir

        Returns:
            [np.ndarray]: Array transformado
        """
        X_std = (X - self.min) / (self.max - self.min)
        X = X_std * (self.Xmax - self.Xmin) + self.Xmin
        return X