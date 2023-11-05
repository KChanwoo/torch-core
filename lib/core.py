import os

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lib.EarlyStopping import EarlyStopping
from lib.score import Scorer


class Core:
    def __init__(self, base_path: str, model: torch.nn.Module, optimizer: Optimizer, loss: torch.nn.Module,
                 scorer: Scorer = None, early_stopping: bool = True):
        self._model = model
        self._base_path = base_path
        self._optimizer = optimizer
        self.__early_stopping = early_stopping
        self._loss = loss
        self._scorer = scorer if scorer is not None else Scorer(self._base_path)

        if not os.path.exists(base_path):
            os.makedirs(base_path)

    def __default_collate(self, batch):
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

    def train(self, dataset: Dataset, dataset_val: Dataset, batch_size=64, num_epochs=1000, collate_fn=None):
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=collate_fn if collate_fn is not None else self.__default_collate)
        val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
                                    collate_fn=collate_fn if collate_fn is not None else self.__default_collate)
        dataloaders_dict = {
            "train": train_dataloader,
            "val": val_dataloader
        }

        save_path = os.path.join(self._base_path, 'weight_train.pt')

        early_stopping = EarlyStopping(patience=10, verbose=True, path=save_path) if self.__early_stopping else None

        device = torch.device(
            "cuda:0" if torch.has_cuda and torch.cuda.is_available() else "mps" if torch.has_mps and torch.mps.is_available() else "cpu")
        print("using device：", device)

        self._model.to(device)

        for epoch in range(num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-------------')
            for phase in ['train', 'val']:
                if phase == 'train':
                    self._model.train()  # set network 'train' mode
                else:
                    self._model.eval()  # set network 'val' mode

                # Before training
                if (epoch == 0) and (phase == 'train'):
                    continue

                # batch loop
                for inputs, labels in tqdm(dataloaders_dict[phase]):
                    # send data to GPU
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # initialize optimizer
                    self._optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self._model(inputs)
                        loss = self._loss(outputs, labels)  # calculate loss
                        # _, preds = torch.max(reshaped, 1)  # predict
                        # back propagtion
                        if phase == 'train':
                            loss.backward()
                            self._optimizer.step()

                        self._scorer.add_batch_result(outputs, labels, loss)

                print("{} result:".format(phase))
                epoch_loss = self._scorer.get_epoch_result(phase == 'val', phase == 'val')

                if phase == 'val':
                    # check early stopping
                    early_stopping(epoch_loss, self._model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        self._scorer.draw_total_result()
