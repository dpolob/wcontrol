import pathlib
import numpy as np
import pandas as pd
import torch
#from torch_lr_finder import LRFinder
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.switch_backend('agg')
#from tqdm.notebook import tqdm   # FOR SCRIP


class TorchTrainer():
    def __init__(self, name, model, optimizer=None, loss_fn=None, scheduler=None, device='cpu', **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.name = name
        self.alpha = kwargs.get('alpha', None)
                
    def _step_optim(self):
        if type(self.optimizer) is list:
            for i in range(len(self.optimizer)):
                self.optimizer[i].step()
                self.optimizer[i].zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
    def _loss_batch(self, Xf, X, Yt, Y, P,  optimize, return_ypred=False, weights: list = None):
        Xf = Xf.to(self.device)
        X = X.to(self.device)
        Yt = Yt.to(self.device)
        Y = Y.to(self.device)
        P = P.to(self.device)
        
        y_pred = self.model(x_f=Xf, x=X, y_t=Yt, y=Y, p=P)
        # definir la perdida para cada componente y debo quitar la componente 0 de Y
        loss_temp = self.loss_fn(y_pred[:, :, 0], Y[:, :-1, 0])
        loss_hr = self.loss_fn(y_pred[:, :, 1], Y[:, :-1:, 1])
        weights = torch.tensor([1/150541, 1/1621, 1/512, 1/249, 1/176, 1/121, 1/46, 1/1]).to(self.device)
        clas_input = y_pred[:, :, 2:].reshape(-1, 8)  # y_pred(N, 72, 8) -> reshape(-1, 8) -> y_pred(N*72, 8), al final compara clases
        clas_target = torch.argmax(Y[:, :-1, 2:], dim=-1).reshape(-1, 1).squeeze().type(torch.long)  # Y(N, 72, 8) -> argmax(dim=-1) -> (N, 72) -> reshape(-1,1) -> (N*72, 1) -> squeeze() -> (N*72)
        loss_class_precipitacion = nn.CrossEntropyLoss(weight=weights)(clas_input, clas_target)
        loss = (1 - self.alpha) * (loss_temp + loss_hr) + self.alpha * loss_class_precipitacion 
        losses = [loss_temp, loss_hr, loss_class_precipitacion]
        
        if optimize:
            loss.backward()
            self._step_optim()
        
        loss_value = loss.item()
        del (Xf, X, Yt, Y, P, loss, losses)
        return loss_value
    
    def evaluate(self, dataloader):
        self.model.eval()
        loss_values = []
        with torch.no_grad():
            for Xf, X, Yt, Y, P in dataloader:
                loss_value = self._loss_batch(Xf, X, Yt, Y, P, optimize=False)
                loss_values.append(loss_value)
        
        loss_value = np.mean(loss_values)
        return loss_value
      
    def train(self, epochs, train_dataloader, valid_dataloader):
        start_epoch = 0
        for i_epoch in range(start_epoch, start_epoch + epochs):
            self.model.train()
            running_loss = 0
            
            for it, (Xf, X, Yt, Y, P) in enumerate(train_dataloader):
                loss = self._loss_batch(Xf=Xf,
                                        X=X,
                                                Yt=Yt,
                                                Y=Y,
                                                P=P,
                                                optimize=True)
                running_loss += loss
                
        if valid_dataloader is not None:
                valid_loss, additional_metrics = self.evaluate(valid_dataloader)
                self.writer.add_scalar('validation loss', valid_loss, i_epoch)
                if additional_metrics is not None:
                    tqdm.write(additional_metrics)
                tqdm.write(f'Valid loss at epoch {i_epoch + 1}- {valid_loss}')
                self.valid_losses[i_epoch + 1] = valid_loss
                #tqdm.write(f"{self.valid_losses[i+1]}")
        
                
            