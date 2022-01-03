import pathlib 
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle

    
def save_dict(path, _dict):
    with open(path, 'wb') as handle:
        pickle.dump(_dict, handle)

class TorchTrainer():
    def __init__(self, model, optimizer=None, loss_fn=None, device='cpu', **kwargs):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer = optimizer

        self.checkpoint_path = Path(kwargs.get('checkpoint_folder', None))
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.runs_path = Path(kwargs.get('runs_folder', None))
        self.runs_path.mkdir(parents=True, exist_ok=True)
        self.train_checkpoint_interval = kwargs.get('train_checkpoint_interval', 1)
        self.max_checkpoints = kwargs.get('max_checkpoints', 50)
        self.writer = None  # se inicializa en el train, para predict no hay 
        self.save_model = kwargs.get('save_model', False)
        self.save_model_path = kwargs.get('save_model_path', None)
        self.early_stop = kwargs.get('early_stop', None)
        if self.save_model:
            torch.save(self.model, pathlib.Path(self.save_model_path))
            
        if (self.checkpoint_path / 'valid_losses.pickle').is_file():
            self.valid_losses = pickle.load(open(self.checkpoint_path/'valid_losses.pickle', 'rb'))
        else:
            self.valid_losses = {}
    
    def _early_stopping(self, epoch_without_resuming: int) -> bool:
        """ Devuelve True si se cumplen las condiciones de early stopping
        ultimo epoch - mejor_epoch > valor definido en self.early_stop"""
        last_epoch_index = len(self.valid_losses)
                
        if last_epoch_index > self.early_stop and epoch_without_resuming > self.early_stop:
            best_epoch_index = sorted(self.valid_losses.items(), key=lambda x:x[1])[0][0]
            if (last_epoch_index - best_epoch_index) > self.early_stop:
                return True
        return False
            
    def _get_checkpoints(self, name=None):
        checkpoints = []
        for cp in self.checkpoint_path.glob('checkpoint_*'):
            checkpoint_name = str(cp).split('/')[-1]
            checkpoint_epoch = int(checkpoint_name.split('_')[-1])
            checkpoints.append((cp, checkpoint_epoch))
        checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
        return checkpoints

    def _clean_outdated_checkpoints(self):
        checkpoints = self._get_checkpoints()
        if len(checkpoints) > self.max_checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
            for delete_cp in checkpoints[self.max_checkpoints:]:
                delete_cp[0].unlink()
                tqdm.write(f'removed checkpoint of epoch - {delete_cp[1]}')

    def _save_checkpoint(self, epoch, valid_loss=None):
        self._clean_outdated_checkpoints()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': [o.state_dict() for o in self.optimizer] if type(self.optimizer) is list else self.optimizer.state_dict(),
        }
        
        if valid_loss:
            checkpoint.update({'loss': valid_loss})
        torch.save(checkpoint, self.checkpoint_path/f'checkpoint_{epoch}')
        save_dict(self.checkpoint_path / 'valid_losses.pickle', self.valid_losses)
        tqdm.write(f'saved checkpoint for epoch {epoch}')
        self._clean_outdated_checkpoints()

    def _load_checkpoint(self, epoch=None, only_model=False, name=None):
        if name is None:
            checkpoints = self._get_checkpoints()
        else:
            checkpoints = self._get_checkpoints(name)
        if len(checkpoints) > 0:
            if not epoch:
                checkpoint_config = checkpoints[0]
            else:
                checkpoint_config = list(filter(lambda x: x[1] == epoch, checkpoints))[0]
            checkpoint = torch.load(checkpoint_config[0])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if not only_model:
                if type(self.optimizer) is list:
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(checkpoint['optimizer_state_dict'][i])
                else:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler is not None:
                    if type(self.scheduler) is list:
                        for i in range(len(self.scheduler)):
                            self.scheduler[i].load_state_dict(checkpoint['scheduler_state_dict'][i])
                    else:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            tqdm.write(f'loaded checkpoint for epoch - {checkpoint["epoch"]}')
            return checkpoint['epoch']
        return None

    def _load_best_checkpoint(self):
        if (self.checkpoint_path/'valid_losses.pickle').is_file():
            best_epoch = sorted(self.valid_losses.items(), key=lambda x:x[1])[0][0]
            loaded_epoch = self._load_checkpoint(epoch=best_epoch, only_model=True)
        else:
            tqdm.write(f"No se ha encontrado el archivo valid_losses")
            exit()

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
        
        y_pred = self.model(input_seq=X)
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
                y_pred = self.model(X, Y)
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
                    # print(f"{torch.argmax(X, dim=-1).cpu().detach().numpy().reshape(-1,1).shape=}")
                    # print(f"{torch.argmax(Y, dim=-1).cpu().detach().numpy().reshape(-1,1).shape=}")
                    # print(f"{torch.argmax(y_pred, dim=-1).cpu().detach().numpy().reshape(-1,1).shape=}")
                    
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
            