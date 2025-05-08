import os

import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torch.utils.data import Dataset

from torchc.core import Core
from torchc.score import MSEScorer


# ---------- synthetic data generation ----------
np.random.seed(42)
n_samples = 5000

# Features: columns 0,1 and 3,4,5 will be used (total 5)
x_01 = np.random.randn(n_samples, 2)
x_dummy = np.random.randn(n_samples, 1)           # column 2 (ignored)
x_345 = np.random.randn(n_samples, 3)

# Target: linear combination + noise
y = 0.3 * x_01.sum(axis=1) + 0.2 * x_345.sum(axis=1) + np.random.randn(n_samples) * 0.05

# Assemble dataset shape (N, 7) -> cols 0..5 + target in col 6
data_array = np.hstack([x_01, x_dummy, x_345, y.reshape(-1, 1)])

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
scorer = MSEScorer("./result/regression")


class MusicDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return np.append(self.dataset[index, :2], self.dataset[index, 3:6]).astype(np.float32), np.array([self.dataset[index, -1]]).astype(np.float32)

    def __len__(self):
        return len(self.dataset)


train_len = int(len(data_array) * 0.9)
train_dataset = MusicDataset(data_array[:train_len])
test_dataset = MusicDataset(data_array[train_len:])

core = Core("./result/regression", model, optimizer, loss, None, scorer, early_stopping=True, early_stopping_skip=10)
core.train(train_dataset, test_dataset, num_epochs=10)
core.test(test_dataset)
