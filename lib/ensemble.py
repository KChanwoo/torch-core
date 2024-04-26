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

    def _vote_hard_un(self, output_all):
        result = []

        for outputs in output_all:
            result_one = self.proceed_outputs(outputs)
            result.append(result_one)

        result = torch.stack(result)
        result = torch.sum(result, dim=0)
        result = (result == len(self.models)).float()
        result = self.add_decay(result)

        return result

    def _vote_hard_maj(self, output_all):
        result = []
        shape = output_all[0].shape
        if len(shape) == 1:
            num_class = 1
        else:
            num_class = shape[1]

        for outputs in output_all:
            result_one = self.proceed_outputs(outputs)
            result.append(result_one)

        result = torch.stack(result)
        result = torch.sum(result, dim=0)
        if num_class > 1:
            result = torch.nn.functional.softmax(result.float())
        else:
            result = (result > len(self.models) / 2).float()
            result = self.add_decay(result)

        return result

    def _vote_soft(self, output_all):

        result = torch.stack(output_all)
        result = torch.mean(result, dim=0)

        return result

    def proceed_outputs(self, output):
        output = output.squeeze()
        shape = output.shape

        if len(shape) == 1:
            # binary
            return (output > .5).float()
        else:
            # multi-class
            args = torch.argmax(output, dim=1)
            return torch.nn.functional.one_hot(args, shape[1])

    def add_decay(self, result):
        decay = 1 - torch.randint_like(result, low=1, high=5) / 100

        return result * decay

    def vote(self, output_all):
        result = []
        if self.mode == VoteEnsemble.HARD_UNANIMOUS:
            result = self._vote_hard_un(output_all)
        elif self.mode == VoteEnsemble.HARD_MAJORITY:
            result = self._vote_hard_maj(output_all)
        elif self.mode == VoteEnsemble.SOFT:
            result = self._vote_soft(output_all)

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

    def _dataset_loop(self, test_dataloader):
        device = torch.device(self._device)

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
            loss_all = .0
            # test ensemble
            with torch.set_grad_enabled(False):
                for i, model in enumerate(self.models):
                    weight = 1. if self.weight is None else self.weight[i]
                    outputs, loss = model.train_step(model.get_model(), inputs, labels, False)
                    output_all.append(outputs * weight)
                    loss_all += loss.item()
                outputs = self.vote(output_all)
                self._scorer.add_batch_result(outputs, labels, loss_all / len(self.models))

    def _datasets_loop(self, batch_size, test_dataset_list, collate_fn_list):
        assert len(self.models) == len(test_dataset_list), \
            "Dataset list must have same length as models: models:{0}, datasets:{1}".format(len(self.models),
                                                                                            len(test_dataset_list))
        device = torch.device(self._device)

        output_all = []
        label_all = []
        loss_all = .0
        for i in range(len(self.models)):
            model = self.models[i]
            test_dataset = test_dataset_list[i]
            collate_fn = None if collate_fn_list is None else collate_fn_list[i]
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=collate_fn if collate_fn is not None else self._default_collate)
            model.load(model.save_path)
            model.get_model().to(device)
            model.get_model().eval()
            loss_one_model = .0
            output_one_model = []
            label_one_model = []
            # batch loop
            for inputs, labels in tqdm(test_dataloader):
                # send data to CPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                # test ensemble
                with torch.set_grad_enabled(False):
                    weight = 1. if self.weight is None else self.weight[i]

                    outputs, loss = model.train_step(model.get_model(), inputs, labels, False)
                    output_one_model.append(outputs.cpu() * weight)
                    label_one_model.append(labels.cpu())
                    loss_one_model += loss.item() * inputs.shape[0]

            loss_all += loss_one_model / len(test_dataset)
            output_all.append(torch.cat(output_one_model, dim=0))
            if len(label_all) == 0:
                label_all.append(torch.cat(label_one_model, dim=0))

        outputs = self.vote(output_all)
        self._scorer.add_batch_result(outputs, label_all[0], loss_all / len(self.models))

    def test(self, test_dataset, batch_size=64, collate_fn=None, test_all=True):
        # test each models
        if test_all:
            for model in self.models:
                model.test(test_dataset, batch_size, collate_fn)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                      collate_fn=collate_fn if collate_fn is not None else self._default_collate)
        self._dataset_loop(test_dataloader)

        self._scorer.get_epoch_result(True, True, True, "test")
        self._scorer.draw_total_result()

    def predict_dataset(self, test_dataset, batch_size=64, collate_fn=None):
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=collate_fn if collate_fn is not None else self._default_collate)

        self._dataset_loop(test_dataloader)

        pred = copy.deepcopy(self._scorer.get_preds())
        output_list = copy.deepcopy(self._scorer.get_outputs())
        self._scorer.reset_epoch()

        return output_list, pred

    def test_datasets(self, test_dataset_list: list[Dataset], batch_size=64, collate_fn_list: Union[list, None]=None):
        self._datasets_loop(batch_size, test_dataset_list, collate_fn_list)

        self._scorer.get_epoch_result(True, True, True, "test")

    def predict_datasets(self, test_dataset_list: list[Dataset], batch_size=64, collate_fn_list: Union[list, None] = None):
        self._datasets_loop(batch_size, test_dataset_list, collate_fn_list)

        pred = copy.deepcopy(self._scorer.get_preds())
        output_list = copy.deepcopy(self._scorer.get_outputs())
        self._scorer.reset_epoch()

        return output_list, pred
