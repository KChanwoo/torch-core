import os

from module.common import LinearModule

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandas as pd
import torch.nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from lib.core import Core
from lib.score import BinaryScorer


"""
Example of binary classification using ECG dataset
https://www.kaggle.com/datasets/shayanfazeli/heartbeat/
"""

normal = pd.read_csv("./data/ptbdb_normal.csv")
abnormal = pd.read_csv("./data/ptbdb_abnormal.csv")


class ECGDataset(Dataset):
    def __init__(self, normals, abnormals):
        self.dataset = []
        self.dataset.extend([{'data': [float(ecg) for ecg in item], 'label': 0} for item in normals])
        self.dataset.extend([{'data': [float(ecg) for ecg in item], 'label': 1} for item in abnormals])

    def __getitem__(self, index) -> T_co:
        return [self.dataset[index]['data']], [float(self.dataset[index]['label'])]

    def __len__(self):
        return len(self.dataset)


class Extractor(torch.nn.Module):
    def forward(self, x):
        return x[0]


model = torch.nn.Sequential(
    torch.nn.LSTM(188, 256, bidirectional=True),
    Extractor(),
    LinearModule(512, 256, activation=torch.nn.ReLU()),
    LinearModule(256, 1, activation=torch.nn.ReLU()),
    torch.nn.Flatten(1),
    LinearModule(1, 1, activation=torch.nn.Softmax())
)

opt = torch.optim.SGD(params=model.parameters(), lr=1.0e-4)
loss = torch.nn.BCELoss()
scorer = BinaryScorer("./result")
core = Core("./result", model, opt, loss, scorer)

normal_train_len = round(len(normal.values) * .9)
abnormal_train_len = round(len(abnormal.values) * .9)

train_dataset = ECGDataset(normal.values[:normal_train_len], abnormal.values[:abnormal_train_len])
val_dataset = ECGDataset(normal.values[normal_train_len: len(normal.values)], abnormal.values[abnormal_train_len: len(abnormal.values)])

core.train(train_dataset, val_dataset, num_epochs=2)
core.test(val_dataset)
