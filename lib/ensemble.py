from typing import Union

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lib.core import Core
from lib.score import Scorer


class VoteEnsemble(Core):
    HARD_UNANIMOUS = 0
    HARD_MAJORITY = 1
    SOFT = 2
    # WEIGHTED = 3

    def __init__(self, base_path: str, models: list[Core], scorer: Scorer = None,
                 mode: Union[HARD_UNANIMOUS, HARD_MAJORITY, SOFT] = SOFT,  # , WEIGHT] = SOFT,
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
                return torch.argmax(output, dim=1)
        else:
            return output

    def vote(self, output_all):
        result = None

        for outputs in output_all:
            squeezed = outputs.squeeze()
            result_one = self.proceed_outputs(squeezed)
            if result is None:
                result = result_one
            else:
                result += result_one

        if self.mode == VoteEnsemble.HARD_UNANIMOUS:
            result = (result == len(self.models)).float()
        elif self.mode == VoteEnsemble.HARD_MAJORITY:
            result = (result == result.max()).float()
        elif self.mode == VoteEnsemble.SOFT:
            result = result / len(self.models)

        return result

    def train(self, dataset: Dataset, dataset_val: Union[Dataset, None], batch_size=64, num_epochs=1000,
              collate_fn=None):
        for model in self.models:
            model.train(dataset, dataset_val, batch_size, num_epochs, collate_fn)

    def test(self, test_dataset, batch_size=64, collate_fn=None, test_all=True):
        # test each models
        if test_all:
            for model in self.models:
                model.test(test_dataset, batch_size, collate_fn)

        train_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=collate_fn if collate_fn is not None else self._default_collate)
        device = torch.device("cpu")

        # batch loop
        for inputs, labels in tqdm(train_dataloader):
            # send data to CPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            output_all = []
            loss_all = torch.tensor(0.0)
            # test ensemble
            with torch.set_grad_enabled(False):
                for model in self.models:
                    model.get_model().to(device)
                    model.get_model().eval()

                    outputs, loss = model.train_step(inputs, labels, False)
                    output_all.append(outputs)
                    loss_all += loss.item()

                outputs = self.vote(output_all)
                self._scorer.add_batch_result(outputs, labels, loss_all / len(self.models))

        self._scorer.get_epoch_result(True, True, True, "test")
        self._scorer.draw_total_result()
