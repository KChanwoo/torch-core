import copy
import os
import random
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

from lib.EarlyStopping import EarlyStopping, EarlyStoppingMode
from lib.score import Scorer, GanScorer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Core:
    def __init__(self, base_path: str, model: Union[None, torch.nn.Module], optimizer: Union[None, Optimizer],
                 loss: Union[None, torch.nn.Module], scheduler=Union[None, torch.optim.lr_scheduler.LRScheduler],
                 scorer: Scorer = None, early_stopping: bool = True, verbose: bool = True, show_fig=False,
                 early_stopping_mode=EarlyStoppingMode.LOSS, use_amp=False):
        self._model = model
        self._base_path = base_path
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.__early_stopping = early_stopping
        self._early_stopping_mode = early_stopping_mode
        self._loss = loss
        self._scorer = scorer if scorer is not None else Scorer(self._base_path, show_fig)
        self._scaler = torch.cuda.amp.GradScaler() if torch.backends.cuda.is_built() and use_amp else None
        self._device = torch.device(
            "cuda:0" if torch.backends.cuda.is_built() else "mps" if torch.backends.mps.is_built() else "cpu")

        self.__verbose = verbose
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        self.save_path = os.path.join(self._base_path, 'weight_train.pt')

    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def train_step(self, model, inputs, labels, train=False):
        # initialize optimizer
        if train:
            self._optimizer.zero_grad()

        if self._scaler is None:
            outputs = model(inputs)
            loss = self._loss(outputs, labels)  # calculate loss

            if train:
                loss.backward()
                self._optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = self._loss(outputs, labels)  # calculate loss

            if train:
                self._scaler.scale(loss).backward()
                self._scaler.step(self._optimizer)
                self._scaler.update()

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
              collate_fn=None, early_stopping=None):
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=collate_fn if collate_fn is not None else self._default_collate)
        val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
                                    collate_fn=collate_fn if collate_fn is not None else self._default_collate) if dataset_val is not None else None
        dataloaders_dict = {
            "train": train_dataloader,
            "val": val_dataloader
        }

        early_stopping = early_stopping if early_stopping is not None else \
            EarlyStopping(patience=5, verbose=True, path=self.save_path,
                          mode=self._early_stopping_mode) if self.__early_stopping else None

        if self._model is not None:
            self._model.to(self._device)

        self.__log("using device：", self._device)
        for epoch in range(num_epochs + 1):
            self.__log('Epoch {}/{}'.format(epoch, num_epochs))
            self.__log('-------------')
            for phase in [key for key in dataloaders_dict.keys()]:
                train = phase == 'train'
                if self._model is not None:
                    if train:
                        self._model.train()  # set network 'train' mode
                    else:
                        self._model.eval()  # set network 'val' mode

                data_loader = dataloaders_dict[phase]
                if data_loader is not None:
                    # batch loop
                    with tqdm(dataloaders_dict[phase]) as pbar:
                        pbar.set_description(f'Epoch: {epoch}')
                        for inputs, labels in pbar:
                            inputs = inputs.to(self._device)
                            labels = labels.to(self._device)

                            # forward
                            with torch.set_grad_enabled(train):
                                outputs, loss = self.train_step(self._model, inputs, labels, train)
                                # _, preds = torch.max(reshaped, 1)  # predict
                                # back propagtion

                                iter_loss = self._scorer.add_batch_result(outputs, labels, loss)

                            pbar.set_postfix(loss=iter_loss, lr=self._optimizer.param_groups[0]['lr'])
                        self.__log("{} result:".format(phase))

                with torch.set_grad_enabled(False):
                    epoch_loss, epoch_acc, epoch_score = self.after_epoch_one(phase)

                if not train:
                    # check early stopping

                    if self._early_stopping_mode == EarlyStoppingMode.LOSS:
                        value = epoch_loss
                    elif self._early_stopping_mode == EarlyStoppingMode.ACCU:
                        value = epoch_acc
                    else:
                        value = epoch_score

                    early_stopping(value,
                                   self._model) if early_stopping is not None and self._model is not None else torch.save(
                        self._model, self.save_path)
                elif self._scheduler is not None:
                    self._scheduler.step()

                if not train and early_stopping is not None and early_stopping.early_stop:
                    self.__log("Early stopping")
                    break

        self._scorer.draw_total_result()

    def dist_train(self, rank: int, world_size: int, dataset: Dataset, dataset_val: Union[Dataset, None], batch_size=64,
                   num_epochs=1000,
                   collate_fn=None, early_stopping=None, seed=42):
        # set same seed to all gpus
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device_id = rank % world_size
        is_main = rank == 0

        self.__verbose = self.__verbose and is_main

        train_sampler = DistributedSampler(dataset, shuffle=False)
        valid_sampler = DistributedSampler(dataset_val, shuffle=False)

        dist_batch_size = int(batch_size / world_size)
        train_dataloader = DataLoader(dataset, batch_size=dist_batch_size, shuffle=False, sampler=train_sampler,
                                      collate_fn=collate_fn if collate_fn is not None else self._default_collate)
        val_dataloader = DataLoader(dataset_val, batch_size=dist_batch_size, shuffle=False, sampler=valid_sampler,
                                    collate_fn=collate_fn if collate_fn is not None else self._default_collate) if dataset_val is not None else None
        dataloaders_dict = {
            "train": train_dataloader,
            "val": val_dataloader
        }

        early_stopping = early_stopping if early_stopping is not None else \
            EarlyStopping(patience=5, verbose=True, path=self.save_path) if self.__early_stopping else None

        model = None
        if self._model is not None:
            self._model.to(device_id)
            model = DDP(self._model, device_ids=[device_id])

        self.__log("using device：", self._device)

        sync_early_stop = torch.tensor(0, device=device_id)
        for epoch in range(num_epochs + 1):
            self.__log('Epoch {}/{}'.format(epoch, num_epochs))
            self.__log('-------------')
            for phase in [key for key in dataloaders_dict.keys()]:
                if sync_early_stop != 0:
                    break

                train = phase == 'train'
                data_loader = dataloaders_dict[phase]
                if model is not None:
                    data_loader.sampler.set_epoch(epoch)
                    if train:
                        model.train()  # set network 'train' mode
                    else:
                        model.eval()  # set network 'val' mode

                if data_loader is not None:
                    # batch loop
                    with tqdm(data_loader, disable=not self.__verbose) as pbar:
                        pbar.set_description(f'Epoch: {epoch}')
                        for inputs, labels in pbar:
                            if sync_early_stop != 0:
                                break
                            inputs = inputs.to(device_id)
                            labels = labels.to(device_id)

                            # forward
                            with torch.set_grad_enabled(train):
                                local_model = model if train else model.module
                                outputs, loss = self.train_step(local_model, inputs, labels, train)
                                # _, preds = torch.max(reshaped, 1)  # predict
                                # back propagtion

                                if train:
                                    output_list = [torch.zeros_like(outputs) for _ in
                                                   range(world_size)] if is_main else None
                                    label_list = [torch.zeros_like(labels) for _ in
                                                  range(world_size)] if is_main else None
                                    loss_list = [torch.zeros_like(loss) for _ in range(world_size)] if is_main else None

                                    dist.gather(outputs, output_list if is_main else None)
                                    dist.gather(labels, label_list if is_main else None)
                                    dist.gather(loss, loss_list if is_main else None)

                                    if is_main:
                                        outputs = torch.concatenate(output_list, dim=0)
                                        labels = torch.concatenate(label_list, dim=0)
                                        loss = torch.mean(torch.stack(loss_list), dim=0)

                                iter_loss = self._scorer.add_batch_result(outputs, labels, loss) if is_main else 0

                            pbar.set_postfix(loss=iter_loss, lr=self._optimizer.param_groups[0]['lr'])

                with torch.set_grad_enabled(False):
                    self.__log("{} result:".format(phase))
                    if is_main:
                        epoch_loss, epoch_acc, epoch_score = self.after_epoch_one(phase)

            if self._scheduler is not None:
                self._scheduler.step()

            if is_main:
                # check early stopping
                early_stopping(epoch_loss,
                               model.module) if early_stopping is not None and model is not None else torch.save(
                    model.module, self.save_path)

                sync_early_stop = torch.tensor(1 if early_stopping is not None and early_stopping.early_stop else 0,
                                               device=device_id)

            # synchronize variable for early stop to all devices
            dist.broadcast(sync_early_stop, 0)

            if sync_early_stop != 0:
                self.__log("Early stopping")
                break

        if is_main:
            self._scorer.draw_total_result()

    def train_hf(self, world_size, config, dataset: Dataset, dataset_val: Union[Dataset, None], batch_size=64,
                 num_epochs=1000):

        try:
            from transformers import TrainingArguments, PreTrainedModel, PretrainedConfig, Trainer, EarlyStoppingCallback
        except:
            raise Exception("need transformers if you want to use train_hf")

        training_args = TrainingArguments(
            output_dir=self._base_path,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=int(batch_size / world_size),
            per_device_eval_batch_size=int(batch_size / world_size),
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            ddp_find_unused_parameters=False
        )

        def compute_metrics(p):
            pred = torch.tensor(p.predictions)
            label = torch.tensor(p.label_ids)
            loss_ = self._loss(pred, label)
            self._scorer.add_batch_result(pred, label, loss_)
            loss__, acc__, f1__ = self.after_epoch_one('val')
            return {'AUC': acc__, 'F1': f1__}

        class CustomModelForSequenceClassification(PreTrainedModel):
            def __init__(self, config, model, loss_fct):
                super().__init__(config)
                self.model = model
                self.loss_fct = loss_fct

            def forward(self, input_ids=None, labels=None):
                logits = self.model(input_ids)
                loss = None
                if labels is not None:
                    loss = self.loss_fct(logits, labels)
                output = (logits,)
                return ((loss,) + output) if loss is not None else output

        callback = []
        if self._early_stopping_mode:
            callback.append(EarlyStoppingCallback(
                early_stopping_patience=5
            ))

        model_hf = CustomModelForSequenceClassification(config, self._model, self._loss)
        trainer = Trainer(
            model=model_hf,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset_val,
            callbacks=callback,
            optimizers=(self._optimizer, self._scheduler),
            compute_metrics=compute_metrics
        )

        trainer.train()

        self._scorer.draw_total_result()

    def test(self, test_dataset, batch_size=64, collate_fn=None):
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn if collate_fn is not None else self._default_collate)
        self.__log("using device：", self._device)

        self._model.to(self._device)
        self.load(self.save_path)
        self._model.eval()  # set network 'val' mode

        # batch loop
        for inputs, labels in tqdm(dataloader, disable=not self.__verbose):
            # send data to GPU
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            # forward
            with torch.set_grad_enabled(False):
                outputs, loss = self.train_step(self._model, inputs, labels, False)
                self._scorer.add_batch_result(outputs, labels, loss)

        self._scorer.get_epoch_result(True, True, True, "test")
        self._scorer.draw_total_result()

    def predict_dataset(self, dataset, batch_size=64, collate_fn=None):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn if collate_fn is not None else self._default_collate)
        self.__log("using device：", self._device)

        self._model.to(self._device)
        self.load(self.save_path)
        self._model.eval()  # set network 'val' mode

        # batch loop
        for inputs, labels in tqdm(dataloader, disable=not self.__verbose):
            # send data to GPU
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)

            # forward
            with torch.set_grad_enabled(False):
                outputs, loss = self.train_step(self._model, inputs, labels, False)
                self._scorer.add_batch_result(outputs, labels, loss)

        pred = copy.deepcopy(self._scorer.get_preds())
        output_list = copy.deepcopy(self._scorer.get_outputs())
        self._scorer.reset_epoch()

        return output_list, pred

    def load(self, pt_path):
        self._model.load_state_dict(
            torch.load(pt_path, map_location=self._device))

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
    def flip_coin(chance=.5):
        return np.random.binomial(1, chance)

    def create_real_label(self, num_batch):
        return torch.ones((num_batch,) + self._output_shape, device=self._device)

    def create_fake_label(self, num_batch):
        return torch.zeros((num_batch,) + self._output_shape, device=self._device)

    def create_seed(self, num_batch):
        return torch.randn(num_batch, *self._seed_shape, device=self._device, dtype=torch.float32)

    def train_step(self, model, inputs, labels, train=False):
        num_batch = inputs.shape[0]
        seed = self.create_seed(num_batch)
        data_gan = self._model_g(seed)

        self._model_g.get_model().eval()

        if GanCore.flip_coin():
            data = inputs
            label = self.create_real_label(num_batch)
        else:
            data = data_gan.detach()
            label = self.create_fake_label(num_batch)

        self._model_d.get_model().train()
        self._model_d.train_step(self._model_d, data, label, train)

        if GanCore.flip_coin(.9):
            label_gan = self.create_real_label(num_batch)
        else:
            label_gan = self.create_fake_label(num_batch)

        if train:
            self._model_g.get_model().zero_grad()

        outputs, loss = self._model_d.train_step(self._model_d, data_gan, label_gan, False)
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
