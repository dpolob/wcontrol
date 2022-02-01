import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from common.utils.trainers.base_trainer import BaseTrainer 

plt.switch_backend('agg')

class TorchTrainer(BaseTrainer):
    def __init__(self, model, optimizer=None, loss_fn=None, scheduler=None, device='cpu', **kwargs):
        super().__init__(model, loss_fn, device, **kwargs)
        
        self.scheduler_batch_step = kwargs.get('scheduler_batch_step', False)
        self.additional_metric_fns = kwargs.get('additional_metric_fns', {})
        self.additional_metric_fns = self.additional_metric_fns.items()
        self.alpha = kwargs.get('alpha', None)
        self.optimizer = optimizer
        self.scheduler = scheduler

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
        
    def _loss_batch(self, Xf, X, Yt, Y, P,  optimize, unitary_metrics: list=None, return_ypred=False, weights: list = None):
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
        del (Xf, X, Yt, Y, P, loss)
    
        if unitary_metrics is not None:
            unitary_metrics = [_.item() for _ in losses]
            if return_ypred:
                return loss_value, unitary_metrics, y_pred
            else:
                return loss_value, unitary_metrics
        else:
            if return_ypred:
                return loss_value, y_pred
            else:
                return loss_value
        
    def evaluate(self, dataloader):
        self.model.eval()
        eval_bar = tqdm(dataloader, leave=False)
        loss_values = []
        with torch.no_grad():
            for Xf, X, Yt, Y, P in eval_bar:
                loss_value = self._loss_batch(Xf, X, Yt, Y, P, optimize=False)
                loss_values.append(loss_value)
        loss_value = np.mean(loss_values)
        eval_bar.set_description("evaluation loss %.2f" % loss_value)
        return loss_value, None
    
    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for Xf, X, Yt, Y, P in tqdm(dataloader):
                Xf = Xf.to(self.device)
                X = X.to(self.device)
                Yt = Yt.to(self.device)
                Y = Y.to(self.device)
                P = P.to(self.device)
                y_pred = self.model(Xf, X, Yt, Y, P)
                predictions.append(y_pred.cpu().numpy())
        return predictions

    def predict_one(self, Xf, X, Yt, P):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            Xf = Xf.to(self.device)
            Yt = Yt.to(self.device)
            P = P.to(self.device)
            y_pred = self.model(Xf, Xf, Yt, Yt, P)
            predictions.append(y_pred.cpu().numpy())
        return predictions    

    
    def train(self, epochs, train_dataloader, valid_dataloader=None, resume=True, resume_only_model=False, plot=False):
        self.writer = SummaryWriter(self.runs_path)
        start_epoch = 0
        epoch_without_resuming = 0
        unitary_metrics = ['loss_temperatura', 'loss_hr', 'loss_precipitacion']
        if resume:
            loaded_epoch = self._load_checkpoint(only_model=resume_only_model)
            if loaded_epoch:
                start_epoch = loaded_epoch
        for i_epoch, epoch_without_resuming in zip(tqdm(range(start_epoch, start_epoch + epochs), leave=True), range(0, epochs)):
            self.model.train()
            training_losses = []
            running_loss = 0
            unitary_running_loss = [0.0 for _ in range(len(unitary_metrics))]

            training_bar = tqdm(train_dataloader, leave=False)
            for it, (Xf, X, Yt, Y, P) in enumerate(training_bar):
                loss, unitary_losses, y_pred = self._loss_batch(Xf=Xf,
                                                X=X,
                                                Yt=Yt,
                                                Y=Y,
                                                P=P,
                                                optimize=True,
                                                unitary_metrics=unitary_metrics,
                                                return_ypred = True)
                running_loss += loss
                for idx, unitary_loss in enumerate(unitary_losses):
                    unitary_running_loss[idx] += unitary_loss

                training_bar.set_description("loss %.4f" % loss)
                if it % 100 == 99:
                    self.writer.add_scalar('training loss', running_loss / 100, i_epoch * len(train_dataloader) + it)
                    for idx, name in enumerate(unitary_metrics):
                        self.writer.add_scalar(name, unitary_running_loss[idx] / 100, i_epoch * len(train_dataloader) + it )    
                    
                    training_losses.append(running_loss / 100)
                    running_loss = 0
                    unitary_running_loss = [0.0 for _ in range(len(unitary_metrics))]
                if plot and it % 1000 == 999:
                    fig =plt.figure(figsize=(13,6))
                    # pintar temp idx =0
                    plt.plot(torch.mean(Y[:, :-1, 0], dim=0).cpu().detach().numpy().reshape(-1,1), 'green')
                    plt.plot(torch.mean(y_pred[..., 0], dim=0).cpu().detach().numpy().reshape(-1,1), 'red')
                    plt.plot(torch.mean(P[:, :-1, 0], dim=0).cpu().detach().numpy().reshape(-1,1), 'magenta')
                    self.writer.add_figure(f"data/test/resultados_temp", fig, i_epoch * len(train_dataloader) + it)
                    fig =plt.figure(figsize=(13,6))
                    # pintar hr idx = 1
                    plt.plot(torch.mean(Y[:, :-1, 1], dim=0).cpu().detach().numpy().reshape(-1,1), 'green')
                    plt.plot(torch.mean(y_pred[..., 1], dim=0).cpu().detach().numpy().reshape(-1,1), 'red')
                    plt.plot(torch.mean(P[:, :-1, 1], dim=0).cpu().detach().numpy().reshape(-1,1), 'magenta')
                    self.writer.add_figure(f"data/test/resultados_hr", fig, i_epoch * len(train_dataloader) + it)
                    # pintar class idx = 2
                    fig =plt.figure(figsize=(13,6))
                    clas_input = y_pred[:, :, 2:].reshape(-1, 8)  # y_pred(N, 72, 8) -> reshape(-1, 8) -> y_pred(N*72, 8), al final compara clases
                    plt.plot(torch.mean(torch.argmax(Y[:, :-1, 2:], dim=-1).type(torch.float), dim=0).cpu().detach().numpy().reshape(-1,1), 'green')
                    plt.plot(torch.mean(torch.argmax(y_pred[:, :, 2:], dim=-1).type(torch.float), dim=0).cpu().detach().numpy().reshape(-1,1), 'red')
                    plt.plot(torch.mean(torch.argmax(P[:, :-1, 2:], dim=-1).type(torch.float), dim=0).cpu().detach().numpy().reshape(-1,1), 'magenta')
                    self.writer.add_figure(f"data/test/resultados_pre", fig, i_epoch * len(train_dataloader) + it)                    

                if self.scheduler is not None and self.scheduler_batch_step:
                    self._step_scheduler()
            tqdm.write(f'Training loss at epoch {i_epoch + 1} - {np.mean(training_losses)}')
            if valid_dataloader is not None:
                valid_loss, additional_metrics = self.evaluate(valid_dataloader)
                self.writer.add_scalar('validation loss', valid_loss, i_epoch)
                if additional_metrics is not None:
                    tqdm.write(additional_metrics)
                tqdm.write(f'Valid loss at epoch {i_epoch + 1}- {valid_loss}')
                self.valid_losses[i_epoch + 1] = valid_loss
            if self.scheduler is not None and not self.scheduler_batch_step:
                self._step_scheduler(valid_loss)
            if (i_epoch + 1) % self.train_checkpoint_interval == 0:
                self._save_checkpoint(i_epoch + 1)
            if valid_dataloader is not None and self.early_stop is not None and self._early_stopping(epoch_without_resuming):
                tqdm.write("Se ha alcanzado la condicion de early stopping!!!!")
                break
        if self.keep_best_checkpoint:
            self._keep_best_checkpoint()
                