import os

from torch.utils.data import Dataset
from transformers import PretrainedConfig

from module.common import LinearModule
from module.vision import ConvModule

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch.nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from lib.core import Core
from lib.score import MulticlassScorer

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)


model = torch.nn.Sequential(
    ConvModule(1, 64, 2, pool=torch.nn.MaxPool2d(4), activation=torch.nn.ReLU()),
    ConvModule(64, 128, 2, pool=torch.nn.MaxPool2d(4), activation=torch.nn.ReLU()),
    ConvModule(128, 256, 1, activation=torch.nn.ReLU()),
    torch.nn.Flatten(),
    LinearModule(256, 10, activation=torch.nn.Softmax())
)


class Dataset_hf(Dataset):
    def __init__(self, origin_dataset):
        self.dataset = origin_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        return {
            'input_ids': data[0],
            'labels': data[1]
        }


opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
loss = torch.nn.CrossEntropyLoss()

core = Core(".\\result\\classifier", model, opt, loss, None, None)

config = PretrainedConfig(
    num_labels=10,
    # hidden_size=10,
)

core.train_hf(1, config, Dataset_hf(mnist_train), Dataset_hf(mnist_test))
