import os

import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from lib.core import Core
from lib.score import MSEScorer

train_data = pd.read_csv("./data/music_train.csv")

model = torch.nn.Sequential(
    torch.nn.Linear(5, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
    torch.nn.ReLU(),
)

optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
loss = torch.nn.MSELoss()
scorer = MSEScorer(".\\result\\regression")


class MusicDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index) -> T_co:
        return np.append(self.dataset[index, :2], self.dataset[index, 3:6]).astype(np.float32), [float(self.dataset[index, -1])]

    def __len__(self):
        return len(self.dataset)


train_len = int(len(train_data.values) * .9)
train_dataset = MusicDataset(train_data.values[:train_len])
test_dataset = MusicDataset(train_data.values[train_len:])

core = Core(".\\result\\regression", model, optimizer, loss, scorer)
core.train(train_dataset, test_dataset)
core.test(test_dataset)
