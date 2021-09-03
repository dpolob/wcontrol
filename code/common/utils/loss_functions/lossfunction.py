#%%
import torch
import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(self, **kwargs) -> None:
        """Genera funciones de perdida a demanda

        Args:
            loss_fn (str, optional): 'mse' para MSELoss, 'maxL1' para Max(L1Loss),
                                  'L1' para L1Loss, 'sum' para sqrt(MSELoss) + L1Loss
                                   'maxMSE' para Max(MSE).Defaults to 'mse'.
            reduction (str, optional): 'mean' para valor medio, 'sum' para suma
            coefs (tuple, optional): Coeficiones para sum
        """
        
        super(LossFunction, self).__init__()
        self.loss_fn = kwargs.get('loss_fn', 'mse')
        self.reduction = kwargs.get('reduction', 'mean')
        self.coefs = kwargs.get(tuple('coefs'), (0.5, 0.5))
        
        if self.loss_fn == 'mse':
            self.mse = nn.MSELoss(reduction=self.reduction)
        elif self.loss_fn == 'L1':
            self.l1 = nn.L1Loss(reduction=self.reduction)
        elif self.loss_fn == 'maxL1':
            self.l1 = nn.L1Loss(reduction='none')
        elif self.loss_fn == 'maxMSE':
            self.mse = nn.MSELoss(reduction='none')
        elif self.loss_fn == 'sum':
            self.mse = nn.MSELoss(reduction=self.reduction)
            self.l1 = nn.L1Loss(reduction=self.reduction)
        else:
            raise Exception("Loss function no definida")
        
        
    def forward(self, y_pred, y):
        if self.loss_fn == 'mse':
            return self.mse(y_pred, y)
        if self.loss_fn == 'maxL1':
            return torch.max(self.l1(y_pred, y))
        if self.loss_fn == 'L1':
            return self.l1(y_pred, y)
        if self.loss_fn == 'sum':
            return self.coefs[0] * self.l1(y_pred, y) + self.coefs[1] * torch.sqrt(self.mse(y_pred, y))
        if self.loss_fn == 'maxMSE':
            return torch.max(self.mse(y_pred, y))
 
# %%
