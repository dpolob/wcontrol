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
        eval_bar = tqdm(dataloader, leave=False)
        loss_values = []
        with torch.no_grad():
            for X, Y in eval_bar:
                loss_value = self._loss_batch(X, Y, optimize=False, rain=rain)
                loss_values.append(loss_value)
                    
        loss_value = np.mean(loss_values)
        eval_bar.set_description("evaluation loss %.2f" % loss_value)
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
        self.writer = SummaryWriter(self.runs_path)
        start_epoch = 0
        epoch_without_resuming = 0
        if resume:
            loaded_epoch = self._load_checkpoint(only_model=resume_only_model)
            if loaded_epoch:
                start_epoch = loaded_epoch
        for i_epoch, epoch_without_resuming in zip(tqdm(range(start_epoch, start_epoch + epochs), leave=True), range(0, epochs)):
            self.model.train()
            running_loss = 0
            training_losses = []
            training_bar = tqdm(train_dataloader, leave=False)
            for it, (X, Y) in enumerate(training_bar):
                loss, y_pred = self._loss_batch(X=X, Y=Y, optimize=True, return_ypred = True, rain=rain)
                running_loss += loss
                training_losses.append(loss)
                training_bar.set_description("loss %.4f" % loss)
                if it % 100 == 99:
                    self.writer.add_scalar('training loss', running_loss / 100, i_epoch * len(train_dataloader) + it)
                    running_loss = 0
                if plot and not rain and it % 100 == 99:
                    fig =plt.figure(figsize=(13,6))
                    plt.plot(X.cpu().detach().numpy().reshape(-1,1), 'magenta')
                    plt.plot(y_pred.cpu().detach().numpy().reshape(-1,1), 'red')
                    plt.plot(Y.cpu().detach().numpy().reshape(-1,1), 'green')
                    self.writer.add_figure(f"data/test/resultados", fig, i_epoch * len(train_dataloader) + it)
                if plot and rain and it % 100 == 99:
                    fig =plt.figure(figsize=(13,6))
                    plt.plot(torch.argmax(X, dim=-1).cpu().detach().numpy().reshape(-1,1), 'magenta')
                    plt.plot(torch.argmax(Y, dim=-1).cpu().detach().numpy().reshape(-1,1), 'green')
                    plt.plot(torch.argmax(y_pred, dim=-1).cpu().detach().numpy().reshape(-1,1), 'red')
                    self.writer.add_figure(f"data/test/resultados", fig, i_epoch * len(train_dataloader) + it)
            tqdm.write(f'Training loss at epoch {i_epoch + 1} - {np.mean(training_losses)}')
            if valid_dataloader is not None:
                valid_loss = self.evaluate(valid_dataloader, rain=rain)
                self.writer.add_scalar('validation loss', valid_loss, i_epoch)
                tqdm.write(f'Valid loss at epoch {i_epoch + 1}- {valid_loss}')
                self.valid_losses[i_epoch + 1] = valid_loss
            if (i_epoch + 1) % self.train_checkpoint_interval == 0:
                self._save_checkpoint(i_epoch + 1)
            if valid_dataloader is not None and self.early_stop is not None and self._early_stopping(epoch_without_resuming):
                tqdm.write("Se ha alcanzado la condicion de early stopping!!!!")
                break
        if self.keep_best_checkpoint:
            self._keep_best_checkpoint()
            
    def predict_one(self) -> Any:
        return super().predict_one()
            