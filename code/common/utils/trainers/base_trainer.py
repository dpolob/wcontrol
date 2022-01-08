from abc import ABC, abstractmethod
import pickle
import pathlib
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from typing import Tuple, Any, List, Union

plt.switch_backend('agg')

class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device='cpu', **kwargs):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_path = kwargs.get('checkpoint_folder', None)
        if self.checkpoint_path is not None: 
            self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.runs_path = kwargs.get('runs_folder', None)
        if self.runs_path is not None: 
            self.runs_path.mkdir(parents=True, exist_ok=True)
        self.train_checkpoint_interval = kwargs.get('train_checkpoint_interval', 1)
        self.max_checkpoints = kwargs.get('max_checkpoints', 50)
        self.writer = None  # se inicializa en el train, para predict no hay 
        self.keep_best_checkpoint = kwargs.get('keep_best_checkpoint', True)
        self.save_model = kwargs.get('save_model', False)
        self.save_model_path = kwargs.get('save_model_path', None)
        self.early_stop = kwargs.get('early_stop', None)
        self.scheduler = kwargs.get('scheduler', None)
        
        if self.save_model:
            torch.save(self.model, pathlib.Path(self.save_model_path))
            
        if self.checkpoint_path is not None:
            if (self.checkpoint_path/'valid_losses.pickle').is_file() and self.checkpoint_path is not None:
                self.valid_losses = pickle.load(open(self.checkpoint_path/'valid_losses.pickle', 'rb')) 
            else:
                self.valid_losses = {}
            
    def _save_dict(self, path: pathlib.Path, _dict: dict) -> None:
        with open(path, 'wb') as handle:
            pickle.dump(_dict, handle)        
    
    def _early_stopping(self, epoch_without_resuming: int) -> bool:
        """ Devuelve True si se cumplen las condiciones de early stopping
        ultimo epoch - mejor_epoch > valor definido en self.early_stop"""
        last_epoch_index = len(self.valid_losses)
                
        if last_epoch_index > self.early_stop and epoch_without_resuming > self.early_stop:
            best_epoch_index = sorted(self.valid_losses.items(), key=lambda x:x[1])[0][0]
            if (last_epoch_index - best_epoch_index) > self.early_stop:
                return True
        return False
            
    def _get_checkpoints(self) -> List[Tuple[pathlib.Path, int]]:
        """ Devuelve la lista de rutas de los archivos "checkpoint_*" y numero de epoch
        Se devuelve ordenado por numero de epoch"""
        
        checkpoints = []
        #checkpoint_path = self.checkpoint_path if name is not None else pathlib.Path(f'./experiments/modelchkpts/{name}_chkpts')
        for cp in self.checkpoint_path.glob('checkpoint_*'):
            checkpoint_name = str(cp).split('/')[-1]
            checkpoint_epoch = int(checkpoint_name.split('_')[-1])
            checkpoints.append((cp, checkpoint_epoch))
        checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
        return checkpoints

    def _clean_outdated_checkpoints(self) -> None:
        """Borra los checkpoint con numero de epoch superior a max_checkpoints"""
        
        checkpoints = self._get_checkpoints()
        if len(checkpoints) > self.max_checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
            for delete_cp in checkpoints[self.max_checkpoints:]:
                delete_cp[0].unlink()
                tqdm.write(f'removed checkpoint of epoch - {delete_cp[1]}')

    def _save_checkpoint(self, epoch: int, valid_loss: float=None) -> None:
        """Almacena el checkpoint y actualiza valid_losses.pickle.
        Borra checkpoints si es necesario"""
        
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
        self._save_dict(self.checkpoint_path / 'valid_losses.pickle', self.valid_losses)
        tqdm.write(f'saved checkpoint for epoch {epoch}')
        self._clean_outdated_checkpoints()

    def _load_checkpoint(self, epoch: int=None, only_model: bool=False) -> Union[None, int]:
        """Carga el checkpoint"""
        
        checkpoints = self._get_checkpoints()
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

    def _load_best_checkpoint(self) -> None:
        """Carga el mejor checkpoint basado en valid_losses.pickle"""
        
        if (self.checkpoint_path/'valid_losses.pickle').is_file():
            best_epoch = sorted(self.valid_losses.items(), key=lambda x:x[1])[0][0]
            self._load_checkpoint(epoch=best_epoch, only_model=True)
        else:
            tqdm.write(f"No se ha encontrado el archivo valid_losses")
            exit()

    def _keep_best_checkpoint(self) -> None:
        """Elimina todos los checkpoints excepto el mejor"""
        
        if self.checkpoint_path/'valid_losses.pickle'.is_file():
            best_epoch = sorted(self.valid_losses.items(), key=lambda x:x[1])[0][0]
            checkpoints = self._get_checkpoints()  # devulve lista ordenada de tuple (path, checkpoint_epoch)
            for delete_cp in checkpoints:
                if delete_cp[1] != best_epoch:
                    delete_cp[0].unlink()
                    
    @abstractmethod
    def _loss_batch(self) -> Any:
        pass
    
    @abstractmethod
    def evaluate(self) -> Any:
        pass
    
    @abstractmethod
    def predict(self) -> Any:
        pass
    
    @abstractmethod
    def evaluate(self) -> Any:
        pass
    
    @abstractmethod
    def predict_one(self) -> Any:
        pass
       
    def train(self) -> Any:
        pass