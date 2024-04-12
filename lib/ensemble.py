import copy
import math
from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from lib.core import Core
from lib.score import Scorer


class VoteEnsemble(Core):
    HARD_UNANIMOUS = 0
    HARD_MAJORITY = 1
    SOFT = 2

    # WEIGHTED = 3

    def __init__(self, base_path: str, models: list[Core], scorer: Scorer = None,
                 mode: int = SOFT,  # , WEIGHT] = SOFT,
                 weight: Union[None, list[float]] = None):
        super().__init__(base_path, None, None, None, scorer=scorer)
        self.models = models
        self.mode = mode
        self.weight = weight
        assert isinstance(self.models, list) and len(self.models) > 1, "You must set multiple models for ensemble"
        # assert mode != VoteEnsemble.WEIGHTED or (
        #         isinstance(weight, list) and len(weight) == len(models)), \
        #     "WEIGHTED mode needs weight value and has same length {0} as model {1}".format(
        #         len(weight) if weight is not None else None, len(models))

    def proceed_outputs(self, output):
        shape = output.shape
        if self.mode == VoteEnsemble.HARD_UNANIMOUS or self.mode == VoteEnsemble.HARD_MAJORITY:
            if len(shape) == 1:
                # binary
                return (output > .5).float()
            else:
                # multi-class
                args = torch.argmax(output, dim=1)
                return torch.nn.functional.one_hot(args, shape[1])
        else:
            return output

    def add_decay(self, result):
        decay = 1 - torch.randint_like(result, low=1, high=5) / 100

        return result * decay

    def vote(self, output_all):
        result = []
        shape = output_all[0].shape
        if len(shape) == 1:
            num_class = 1
        else:
            num_class = shape[1]

        for outputs in output_all:
            squeezed = outputs.squeeze()
            result_one = self.proceed_outputs(squeezed)
            result.append(result_one)

        result = torch.stack(result)
        if self.mode == VoteEnsemble.HARD_UNANIMOUS:
            result = torch.sum(result, dim=0)
            result = (result == len(self.models)).float()
            result = self.add_decay(result)
        elif self.mode == VoteEnsemble.HARD_MAJORITY:
            result = torch.sum(result, dim=0)
            if num_class > 1:
                result = torch.nn.functional.softmax(result.float())
            else:
                result = (result > len(self.models) / 2).float()
                result = self.add_decay(result)
        elif self.mode == VoteEnsemble.SOFT:
            result = torch.mean(result, dim=0)

        return result

    def train(self, dataset: Dataset, dataset_val: Union[Dataset, None], batch_size=64, num_epochs=1000,
              collate_fn=None, bagging=False):

        n_model = len(self.models)
        n_data = 100 / n_model
        if bagging:
            lengths = [(math.ceil(n_data) if i == n_model - 1 else math.floor(n_data)) / 100 for i in range(n_model)]
            subs = random_split(dataset, lengths)
        else:
            subs = [dataset for i in range(n_model)]

        for sub_dataset, model in zip(subs, self.models):
            model.train(sub_dataset, dataset_val, batch_size, num_epochs, collate_fn)

    def test(self, test_dataset, batch_size=64, collate_fn=None, test_all=True):
        # test each models
        if test_all:
            for model in self.models:
                model.test(test_dataset, batch_size, collate_fn)

        train_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                      collate_fn=collate_fn if collate_fn is not None else self._default_collate)
        device = torch.device("cpu")

        for model in self.models:
            model.load(model.save_path)
            model.get_model().to(device)
            model.get_model().eval()

        # batch loop
        for inputs, labels in tqdm(train_dataloader):
            # send data to CPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            output_all = []
            loss_all = torch.tensor(0.0)
            # test ensemble
            with torch.set_grad_enabled(False):
                for i, model in enumerate(self.models):
                    weight = 1. if self.weight is None else self.weight[i]
                    outputs, loss = model.train_step(inputs, labels, False)
                    output_all.append(outputs * weight)
                    loss_all += loss.item()

                outputs = self.vote(output_all)
                self._scorer.add_batch_result(outputs, labels, loss_all / len(self.models))

        self._scorer.get_epoch_result(True, True, True, "test")
        self._scorer.draw_total_result()

    def predict_dataset(self, test_dataset, batch_size=64, collate_fn=None):
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=collate_fn if collate_fn is not None else self._default_collate)
        device = torch.device("cpu")

        for model in self.models:
            model.load(model.save_path)
            model.get_model().to(device)
            model.get_model().eval()

        # batch loop
        for inputs, labels in tqdm(test_dataloader):
            # send data to CPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            output_all = []
            loss_all = torch.tensor(0.0)
            # test ensemble
            with torch.set_grad_enabled(False):
                for i, model in enumerate(self.models):
                    weight = 1. if self.weight is None else self.weight[i]
                    outputs, loss = model.train_step(inputs, labels, False)
                    output_all.append(outputs * weight)
                    loss_all += loss.item()

                outputs = self.vote(output_all)
                self._scorer.add_batch_result(outputs, labels, loss_all / len(self.models))

        pred = copy.deepcopy(self._scorer.get_preds())
        output_list = copy.deepcopy(self._scorer.get_outputs())
        self._scorer.reset_epoch()

        return output_list, pred
