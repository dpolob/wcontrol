import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from common.utils.trainers.base_trainer import BaseTrainer 
from typing import Any

plt.switch_backend('agg')
    
class TorchTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, device='cpu', **kwargs):
        super().__init__(model, loss_fn, device, **kwargs)
        self.optimizer = optimizer

    def _step_optim(self):
        if type(self.optimizer) is list:
            for i in range(len(self.optimizer)):
                self.optimizer[i].step()
                self.optimizer[i].zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _step_scheduler(self, valid_loss=None):
        if type(self.scheduler) is list:
            for i in range(len(self.scheduler)):
                if self.scheduler[i].__class__.__name__ == 'ReduceLROnPlateau':
                    self.scheduler[i].step(valid_loss)
                else:
                    self.scheduler[i].step()
        else:
            if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()
     
    def _loss_batch(self, X, Y,  optimize, return_ypred=False, rain=False):
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        y_pred = self.model(X)
        if rain:
            Y = torch.argmax(Y, dim=-1).reshape(-1, 1).squeeze().type(torch.long)  # Y(1, 72, 8) -> argmax(dim=-1) -> (1, 72) -> reshape(-1,1) -> (72, 1) -> squeeze() -> (72)
            y_pred = y_pred.squeeze()  # y_pred(1, 72, 8) -> squeeze(axis=0) -> (72, 8)
       
        loss = self.loss_fn(y_pred, Y)
                     
        if optimize:
            loss.backward()
            self._step_optim()

        loss_value = loss.item()
        return loss_value if not return_ypred else (loss_value, y_pred)
        
    def evaluate(self, dataloader, rain=False):
        self.model.eval()
        loss_values = []
        with torch.no_grad():
            for X, Y in dataloader:
                loss_value = self._loss_batch(X, Y, optimize=False, rain=rain)
                loss_values.append(loss_value)
                    
        loss_value = np.mean(loss_values)
        return loss_value
    
    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X, Y in tqdm(dataloader):
                X = X.to(self.device)
                Y = Y.to(self.device)
                y_pred = self.model(X)
                predictions.append(y_pred.cpu().numpy())
        return predictions

    def train(self, epochs, train_dataloader, valid_dataloader=None, resume=True, resume_only_model=False, plot=False, rain=False):
        start_epoch = 0
        if resume:
            loaded_epoch = self._load_checkpoint(only_model=resume_only_model)
            if loaded_epoch:
                start_epoch = loaded_epoch
        for i_epoch in tqdm(range(start_epoch, start_epoch + epochs)):
            self.model.train()
            running_loss = 0
            training_losses = []
            for it, (X, Y) in enumerate(train_dataloader):
                loss, y_pred = self._loss_batch(X=X, Y=Y, optimize=True, return_ypred = True, rain=rain)
                running_loss += loss
                training_losses.append(loss)
            if valid_dataloader is not None:
                valid_loss = self.evaluate(valid_dataloader, rain=rain)
                self.writer.add_scalar('validation loss', valid_loss, i_epoch)
                self.valid_losses[i_epoch + 1] = valid_loss
           
                   
    def predict_one(self) -> Any:
        return super().predict_one()
            