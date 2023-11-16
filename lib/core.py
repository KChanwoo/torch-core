import os
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lib.EarlyStopping import EarlyStopping
from lib.score import Scorer, GanScorer


class Core:
    def __init__(self, base_path: str, model: Union[None, torch.nn.Module], optimizer: Union[None, Optimizer],
                 loss: Union[None, torch.nn.Module],
                 scorer: Scorer = None, early_stopping: bool = True, verbose: bool = True):
        self._model = model
        self._base_path = base_path
        self._optimizer = optimizer
        self.__early_stopping = early_stopping
        self._loss = loss
        self._scorer = scorer if scorer is not None else Scorer(self._base_path)
        self._device = torch.device(
            "cuda:0" if torch.has_cuda and torch.cuda.is_available() else "mps" if torch.has_mps and torch.mps.is_available() else "cpu")

        self.__verbose = verbose

        self.__log("using device：", self._device)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        self.save_path = os.path.join(self._base_path, 'weight_train.pt')

    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def train_step(self, inputs, labels, train=False):
        # initialize optimizer
        if train:
            self._optimizer.zero_grad()

        outputs = self._model(inputs)
        loss = self._loss(outputs, labels)  # calculate loss

        if train:
            loss.backward()
            self._optimizer.step()
        return outputs, loss

    def _default_collate(self, batch):
        item = batch[0]
        img = item[0]
        lbl = item[1]
        if isinstance(img, Tensor):
            image = torch.stack([item[0] for item in batch])
        else:
            image = torch.tensor([item[0] for item in batch])

        if isinstance(lbl, Tensor):
            label = torch.stack([item[1] for item in batch])
        else:
            label = torch.tensor([item[1] for item in batch])

        return image, label

    def after_epoch_one(self, phase):
        return self._scorer.get_epoch_result(phase == 'val', phase == 'val')

    def train(self, dataset: Dataset, dataset_val: Union[Dataset, None], batch_size=64, num_epochs=1000,
              collate_fn=None):
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=collate_fn if collate_fn is not None else self._default_collate)
        val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
                                    collate_fn=collate_fn if collate_fn is not None else self._default_collate) if dataset_val is not None else None
        dataloaders_dict = {
            "train": train_dataloader,
            "val": val_dataloader
        }

        early_stopping = EarlyStopping(patience=10, verbose=True, path=self.save_path) if self.__early_stopping else None

        if self._model is not None:
            self._model.to(self._device)

        for epoch in range(num_epochs + 1):
            self.__log('Epoch {}/{}'.format(epoch, num_epochs))
            self.__log('-------------')
            for phase in [key for key in dataloaders_dict.keys()]:
                if self._model is not None:
                    if phase == 'train':
                        self._model.train()  # set network 'train' mode
                    else:
                        self._model.eval()  # set network 'val' mode

                # Before training
                if (epoch == 0) and (phase == 'train'):
                    continue

                data_loader = dataloaders_dict[phase]
                if data_loader is not None:
                    # batch loop
                    for inputs, labels in tqdm(dataloaders_dict[phase]):
                        inputs = inputs.to(self._device)
                        labels = labels.to(self._device)

                        # forward
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs, loss = self.train_step(inputs, labels, phase == 'train')
                            # _, preds = torch.max(reshaped, 1)  # predict
                            # back propagtion
                            self._scorer.add_batch_result(outputs, labels, loss)

                    self.__log("{} result:".format(phase))

                with torch.set_grad_enabled(False):
                    epoch_loss = self.after_epoch_one(phase)

                if phase == 'val':
                    # check early stopping
                    early_stopping(epoch_loss, self._model) if early_stopping is not None and self._model is not None else 0

            if early_stopping is not None and early_stopping.early_stop:
                self.__log("Early stopping")
                break

        self._scorer.draw_total_result()

    def test(self, test_dataset, batch_size=64, collate_fn=None):
        train_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=collate_fn if collate_fn is not None else self._default_collate)
        device = torch.device(
            "cuda:0" if torch.has_cuda and torch.cuda.is_available() else "mps" if torch.has_mps and torch.mps.is_available() else "cpu")
        self.__log("using device：", device)

        self._model.to(device)
        self._model.load_state_dict(torch.load(self.save_path, map_location=device))
        self._model.eval()  # set network 'val' mode

        # batch loop
        for inputs, labels in tqdm(train_dataloader):
            # send data to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            with torch.set_grad_enabled(False):
                outputs, loss = self.train_step(inputs, labels, False)
                self._scorer.add_batch_result(outputs, labels, loss)

        self._scorer.get_epoch_result(True, True, True, "test")
        self._scorer.draw_total_result()

    def load(self, pt_path):
        self._model.load_state_dict(
            torch.load(pt_path, map_location="cuda" if torch.has_cuda and torch.cuda.is_available() else "cpu"))

    def save(self, pt_path):
        torch.save(self._model.state_dict(), pt_path)

    def get_model(self):
        return self._model

    def get_optim(self):
        return self._optimizer

    def get_criterion(self):
        return self._loss

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
    def __flip_coin(chance=.5):
        return np.random.binomial(1, chance)

    def create_real_label(self, num_batch):
        return torch.ones((num_batch,) + self._output_shape, device=self._device)

    def create_fake_label(self, num_batch):
        return torch.zeros((num_batch,) + self._output_shape, device=self._device)

    def create_seed(self, num_batch):
        return torch.randn(num_batch, *self._seed_shape, device=self._device, dtype=torch.float32)

    def train_step(self, inputs, labels, train=False):
        num_batch = inputs.shape[0]
        seed = self.create_seed(num_batch)
        data_gan = self._model_g(seed)

        self._model_g.get_model().eval()

        if GanCore.__flip_coin():
            data = inputs
            label = self.create_real_label(num_batch)
        else:
            data = data_gan.detach()
            label = self.create_fake_label(num_batch)

        self._model_d.get_model().train()
        self._model_d.train_step(data, label, train)

        if GanCore.__flip_coin(.9):
            label_gan = self.create_real_label(num_batch)
        else:
            label_gan = self.create_fake_label(num_batch)

        if train:
            self._model_g.get_model().zero_grad()

        outputs, loss = self._model_d.train_step(data_gan, label_gan, False)
        loss = self._loss(outputs, label_gan)

        if train:
            loss.backward()
            self._optimizer.step()
        return outputs, loss

    def train(self, dataset: Dataset, dataset_val: Dataset = None, batch_size=64, num_epochs=1000, collate_fn=None):
        self._model_g.get_model().to(self._device)
        self._model_d.get_model().to(self._device)
        super(GanCore, self).train(dataset, None, batch_size, num_epochs, collate_fn)

    def generate(self, number=1):
        seed = self.create_seed(number)
        return self._model_g(seed)
