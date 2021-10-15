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
import pickle

def save_dict(path, _dict):
    with open(path, 'wb') as handle:
        pickle.dump(_dict, handle)

class TorchTrainer():
    def __init__(self, name, model, optimizer=None, loss_fn=None, scheduler=None, device='cpu', **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.name = name
        self.checkpoint_path = kwargs.get('checkpoint_folder', None)
        self.runs_path = kwargs.get('runs_folder', None)
        self.train_checkpoint_interval = kwargs.get('train_checkpoint_interval', 1)
        self.max_checkpoints = kwargs.get('max_checkpoints', 50)
        self.writer = SummaryWriter(self.runs_path)
        
        self.scheduler_batch_step = kwargs.get('scheduler_batch_step', False)
        self.additional_metric_fns = kwargs.get('additional_metric_fns', {})
        self.additional_metric_fns = self.additional_metric_fns.items()
        self.pass_y = kwargs.get('pass_y', False)
        self.save_model = kwargs.get('save_model', False)
        self.save_model_path = kwargs.get('save_model_path', None)
        
        self.early_stop = kwargs.get('early_stop', None)
                
        if self.save_model:
            torch.save(self.model, pathlib.Path(self.save_model_path))
            
        if (self.checkpoint_path/'valid_losses.pickle').is_file():
            self.valid_losses = pickle.load(open(self.checkpoint_path/'valid_losses.pickle', 'rb'))
            #tqdm.write('valid_losses file found!')
        else:
            self.valid_losses = {}
    
    def _early_stopping(self) -> bool:
        """ Devuelve True si se cumplen las condiciones de early stopping
        ultimo epoch - mejor_epoch > valor definido en self.early_stop"""
        
        if isinstance(self.early_stop, int):
            if len(self.valid_losses) > self.early_stop:
                best_epoch_index = sorted(self.valid_losses.items(), key=lambda x:x[1])[0][0]
                last_epoch_index = sorted(self.valid_losses.items(), key=lambda x:x[1])[-1][0]
                if (last_epoch_index - best_epoch_index) > self.early_stop:
                    return True
        return False
            
    def _get_checkpoints(self, name=None):
        checkpoints = []
        #checkpoint_path = self.checkpoint_path if name is not None else pathlib.Path(f'./experiments/modelchkpts/{name}_chkpts')
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
        if self.scheduler is not None:
            checkpoint.update({
                'scheduler_state_dict': [o.state_dict() for o in self.scheduler] if type(self.scheduler) is list else self.scheduler.state_dict()
            })
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
        
    def _loss_batch(self, Xt, X, Yt, Y, teacher, optimize, pass_y, additional_metrics=None, return_ypred=False):
        #if type(xb) is list:
        #    xb = [xbi.to(self.device) for xbi in xb]
        #else:
        #    xb = xb.to(self.device)
        #yb = yb.to(self.device)

        Xt = Xt.to(self.device)
        X = X.to(self.device)
        Yt = Yt.to(self.device)
        Y = Y.to(self.device)
        
        if pass_y:
            y_pred = self.model(Xt, X, Yt, Y, teacher)
        else:
            y_pred = self.model(Xt, X, Yt, Y, teacher)

        loss = self.loss_fn(y_pred, Y[:, 1:, :])  # debo quitar la componente 0
        if additional_metrics is not None:
            additional_metrics = [fn(y_pred, Y[:, 1:, :]) for name, fn in additional_metrics]
        if optimize:
            loss.backward()
            self._step_optim()
        loss_value = loss.item()
        del Xt
        del X
        del Yt
        del Y
        
        del loss
        if additional_metrics is not None:
            if return_ypred:
                return loss_value, additional_metrics, y_pred
            else:
                return loss_value, additional_metrics
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
            for Xt, X, Yt, Y in eval_bar:
                loss_value = self._loss_batch(Xt, X, Yt, Y, False, False, False)
                loss_values.append(loss_value)
                # if len(loss_values[0]) > 1:
                #     loss_value = np.mean([lv[0] for lv in loss_values])
                #     additional_metrics = np.mean([lv[1] for lv in loss_values], axis=0)
                #     additional_metrics_result = {name: result for (name, fn), result in zip(self.additional_metric_fns, additional_metrics)}
                #     return loss_value, additional_metrics_result
                # # eval_bar.set_description("evaluation loss %.2f" % loss_value)
                # else:
        
        loss_value = np.mean(loss_values)
        eval_bar.set_description("evaluation loss %.2f" % loss_value)
        return loss_value, None
    
    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for xt, x, yt, y in tqdm(dataloader):
                # if type(xb) is list:
                #     xb = [xbi.to(self.device) for xbi in xb]
                # else:
                #     xb = xb.to(self.device)
                # yb = yb.to(self.device)
                xt = xt.to(self.device)
                x = x.to(self.device)
                yt = yt.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(xt, x, yt, y)
                #tqdm.write(y_pred.shape)
                predictions.append(y_pred.cpu().numpy())
        #tqdm.write(predictions)
        return predictions

    # pass single batch input, without batch axis
    def predict_one(self, xt, x, yt):
        self.model.eval()
        with torch.no_grad():
            # if type(x) is list:
            #     x = [xi.to(self.device).unsqueeze(0) for xi in x]
            # else:
            #     x = x.to(self.device).unsqueeze(0)
            xt = xt.to(self.device)
            x = x.to(self.device)
            yt = yt.to(self.device)

            y_pred = self.model(x)
            if self.device == 'cuda':
                y_pred = y_pred.cpu()
            y_pred = y_pred.numpy()
            return y_pred
    
    # def lr_find(self, dl, optimizer=None, start_lr=1e-7, end_lr=1e-2, num_iter=200):
    #     if optimizer is None:
    #         optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6, momentum=0.9)
    #     lr_finder = LRFinder(self.model, optimizer, self.loss_fn, device=self.device)
    #     lr_finder.range_test(dl, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter)
    #     lr_finder.plot()
        
    def train(self, epochs, train_dataloader, valid_dataloader=None, resume=True, resume_only_model=False, plot=False):
        start_epoch = 0
        if resume:
            loaded_epoch = self._load_checkpoint(only_model=resume_only_model)
            if loaded_epoch:
                start_epoch = loaded_epoch
        for i in tqdm(range(start_epoch, start_epoch + epochs), leave=True):
            self.model.train()
            training_losses = []
            running_loss = 0
            # additional_running_loss = [0 for _ in self.additional_metric_fns]
            training_bar = tqdm(train_dataloader, leave=False)
            for it, (Xt, X, Yt, Y) in enumerate(training_bar):
                loss, y_pred = self._loss_batch(Xt=Xt,
                                        X=X,
                                        Yt=Yt,
                                        Y=Y,
                                        teacher=True,
                                        optimize=True,
                                        pass_y=self.pass_y,
                                        return_ypred = True)
                running_loss += loss
                # for i, additional_loss in enumerate(additional_metrics):
                #     additional_running_loss[i] += additional_loss

                training_bar.set_description("loss %.4f" % loss)
                if it % 100 == 99:
                    self.writer.add_scalar('training loss', running_loss / 100, i * len(train_dataloader) + it)
                    # for i, additional_loss in enumerate(additional_running_loss):
                    #     self.writer.add_scalar(self.additional_metric_fns.keys[i], additional_loss /100, i * len(train_dataloader) + it )    
                    training_losses.append(running_loss / 100)
                    running_loss = 0
                    # additional_running_loss = [0 for _ in self.additional_metric_fns]
                if plot and it % 10000 == 9999:
                    fig =plt.figure(figsize=(26,12))
                    plt.plot(torch.mean(Y, dim=0).cpu().detach().numpy().reshape(-1,1), 'b')
                    plt.plot(torch.mean(y_pred, dim=0).cpu().detach().numpy().reshape(-1,1), 'r')
                    self.writer.add_figure('data/test/resultados', fig, i * len(train_dataloader) + it)
                    #tqdm.write("Plot en{}".format(i * len(train_dataloader) + it))

                if self.scheduler is not None and self.scheduler_batch_step:
                    self._step_scheduler()
            tqdm.write(f'Training loss at epoch {i + 1} - {np.mean(training_losses)}')
            if valid_dataloader is not None:
                valid_loss, additional_metrics = self.evaluate(valid_dataloader)
                self.writer.add_scalar('validation loss', valid_loss, i)
                if additional_metrics is not None:
                    tqdm.write(additional_metrics)
                tqdm.write(f'Valid loss at epoch {i + 1}- {valid_loss}')
                self.valid_losses[i+1] = valid_loss
                #tqdm.write(f"{self.valid_losses[i+1]}")
            if self.scheduler is not None and not self.scheduler_batch_step:
                self._step_scheduler(valid_loss)
            if (i + 1) % self.train_checkpoint_interval == 0:
                self._save_checkpoint(i+1)
            if valid_dataloader is not None and self.early_stop is not None and self._early_stopping():
                tqdm.write("Se ha alcanzado la condicion de early stopping!!!!")
                break
            