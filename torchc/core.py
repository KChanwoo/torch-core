import copy
import os
from typing import Union

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset

from torchc.lightning import PLDataModule, PLModel, DelayedEarlyStopping
from torchc.score import Scorer, GanScorer


class Core:
    def __init__(self, base_path: str, model: Union[None, torch.nn.Module], optimizer: Union[None, Optimizer],
                 loss: Union[None, torch.nn.Module], scheduler=Union[None, torch.optim.lr_scheduler.LRScheduler],
                 scorer: Scorer = None, early_stopping: bool = True, verbose: bool = True, show_fig=False, use_amp=False, patience=5, early_stopping_skip=0):
        self._device = 'cuda' if torch.backends.cuda.is_built() else 'mps' if torch.backends.mps.is_built() else 'cpu'
        self._accelerator = 'gpu' if torch.backends.cuda.is_built() else 'mps' if torch.backends.mps.is_built() else 'cpu'

        self._model = model
        self._base_path = base_path
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.__early_stopping = early_stopping
        self._loss = loss
        self._scorer = scorer if scorer is not None else Scorer(self._base_path, show_fig)
        self._scaler = torch.cuda.amp.GradScaler() if torch.backends.cuda.is_built() and use_amp else None

        self._patience = patience
        self._early_stopping_skip = early_stopping_skip
        self.__verbose = verbose
        if not os.path.exists(base_path):
            try:
                os.makedirs(base_path)
            except FileExistsError as e:
                pass

        self.save_path = os.path.join(self._base_path, 'weight_train.pt')
        self.save_path_hf = os.path.join(self._base_path, 'model.safetensors')

        self._train_model = PLModel(self)

    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def train_step(self, model, *tensors):
        inputs, labels = tensors[:-1], tensors[-1]
        if self._scaler is None:
            outputs = model(*inputs)
            loss = self._loss(outputs, labels)  # calculate loss
        else:
            with torch.cuda.amp.autocast():
                outputs = model(*inputs)
                loss = self._loss(outputs, labels)  # calculate loss

        return outputs, loss

    def after_epoch_one(self, phase):
        return self._scorer.get_epoch_result(phase == 'val', phase == 'val')

    def train(self, dataset: Dataset, dataset_val: Union[Dataset, None], world_size: int = 1, batch_size=64,
              num_epochs=1000, collate_fn=None, num_workers=1):
        callbacks = [
            ModelCheckpoint(
                monitor='avg_val_loss',
                dirpath=self._base_path,
                filename='weight_train',
                save_top_k=1,
                mode='min'
            )
        ]

        if self.__early_stopping:
            callbacks.append(
                DelayedEarlyStopping(
                    monitor='avg_val_loss',
                    patience=self._patience,
                    verbose=True,
                    mode='min',
                    log_rank_zero_only=True,
                    start_epoch=self._early_stopping_skip
                )
            )

        logger = TensorBoardLogger(self._base_path, name="model_logs")

        trainer = Trainer(
            max_epochs=num_epochs,
            accelerator=self._accelerator,
            devices=world_size,
            callbacks=callbacks,
            logger=logger,
            check_val_every_n_epoch=1 if dataset_val is not None else 0  # validation 주기 설정
        )

        data_module = PLDataModule(train_dataset=dataset, valid_dataset=dataset_val, batch_size=batch_size, num_workers=num_workers,
                                   collate_fn=collate_fn if collate_fn is not None else None)

        trainer.fit(self._train_model, data_module)

    def test(self, test_dataset: Dataset, world_size: int = 1, batch_size: int = 64, collate_fn=None, num_workers=1):
        trainer = Trainer(
            accelerator=self._accelerator,
            devices=world_size,
            logger=False
        )

        self.load()
        self._train_model.eval()

        data_module = PLDataModule(test_dataset=test_dataset, batch_size=batch_size, num_workers=num_workers,
                                   collate_fn=collate_fn if collate_fn is not None else None)

        trainer.test(self._train_model, data_module)

    def predict_dataset(self, dataset: Dataset, world_size: int = 1, batch_size=64, collate_fn=None):
        self.test(dataset, world_size, batch_size, collate_fn)
        pred = copy.deepcopy(self._scorer.get_preds())
        output_list = copy.deepcopy(self._scorer.get_outputs())
        self._scorer.reset_epoch()

        return output_list, pred

    def predict(self, x):
        x = x.to(self._device)
        return self._model(x)

    def load(self, train=False):
        self._train_model = PLModel.load_from_checkpoint(os.path.join(self._base_path, "weight_train.ckpt"), core=self, map_location=self._device)
        self._model = self._train_model.model
        if not train:
            self._model.eval()

    def save(self, pt_path):
        torch.save(self._model.state_dict(), pt_path)

    def get_model(self):
        return self._model

    def get_optim(self):
        return self._optimizer

    def get_criterion(self):
        return self._loss

    def get_scorer(self):
        return self._scorer

    def get_scheduler(self):
        return self._scheduler

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)


class GanCore(Core):
    def __init__(self,
                 base_path: str,
                 model_g: Union[None, Core],
                 model_d: Union[None, Core],
                 optimizer: Optimizer = None,
                 loss: torch.nn.Module = None,
                 scorer: Scorer = None,
                 seed_shape: tuple[int, ...] = (1,),
                 output_shape: tuple[int, ...] = (1,)):

        self._model_g = model_g
        self._model_d = model_d
        self._seed_shape = seed_shape
        self._output_shape = output_shape
        gan_optim = torch.optim.Adam(
            params=self._model_g.get_model().parameters(), lr=2e-4,
            weight_decay=8e-9) if optimizer is None else optimizer
        gan_loss = torch.nn.BCELoss() if loss is None else loss
        scorer = GanScorer(base_path, self, seed_shape) if scorer is None else scorer

        super().__init__(base_path, None, gan_optim, gan_loss, scorer, early_stopping=False)

    @staticmethod
    def flip_coin(chance=.5):
        return np.random.binomial(1, chance)

    def create_real_label(self, num_batch):
        return torch.ones((num_batch,) + self._output_shape, device=self._device)

    def create_fake_label(self, num_batch):
        return torch.zeros((num_batch,) + self._output_shape, device=self._device)

    def create_seed(self, num_batch):
        return torch.randn(num_batch, *self._seed_shape, device=self._device, dtype=torch.float32)

    def train_step(self, model, inputs, labels, d_train=True):
        num_batch = inputs.shape[0]
        seed = self.create_seed(num_batch)

        with torch.set_grad_enabled(False):
            data_gan = self._model_g(seed)

        if d_train:

            if GanCore.flip_coin():
                data = inputs
                label = self.create_real_label(num_batch)
            else:
                data = data_gan.detach()
                label = self.create_fake_label(num_batch)
            outputs, loss = self._model_d.train_step(self._model_d, data, label)
        else:
            if GanCore.flip_coin(.9):
                label_gan = self.create_real_label(num_batch)
            else:
                label_gan = self.create_fake_label(num_batch)

            outputs, loss = self._model_d.train_step(self._model_d, data_gan, label_gan)
            loss = self._loss(outputs, label_gan)

        return outputs, loss

    def train(self, dataset: Dataset, dataset_val: Dataset = None, world_size: int = 1, batch_size=64, num_epochs=1000, collate_fn=None):
        self._model_g.get_model().to(self._device)
        self._model_d.get_model().to(self._device)
        super(GanCore, self).train(dataset, None, batch_size, num_epochs, collate_fn)

    def generate(self, number=1):
        seed = self.create_seed(number)
        return self._model_g(seed)

    def get_g_model(self):
        return self._model_g

    def get_d_model(self):
        return self._model_d
