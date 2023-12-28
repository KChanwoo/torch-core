import os

from lib.ensemble import VoteEnsemble
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


# model 1
model = torch.nn.Sequential(
    ConvModule(1, 64, 2, pool=torch.nn.MaxPool2d(4), activation=torch.nn.ReLU()),
    ConvModule(64, 128, 2, pool=torch.nn.MaxPool2d(4), activation=torch.nn.ReLU()),
    ConvModule(128, 256, 1, activation=torch.nn.ReLU()),
    torch.nn.Flatten(),
    LinearModule(256, 10)
)

opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
loss = torch.nn.CrossEntropyLoss()
scorer = MulticlassScorer(".\\result\\classifier", 10)

core = Core(".\\result\\classifier", model, opt, loss, scorer)

model = torch.nn.Sequential(
    LinearModule(64 * 64, 64 * 64, activation=torch.nn.ReLU()),
    LinearModule(64 * 64, 64 * 64, activation=torch.nn.ReLU()),
    LinearModule(64 * 64, 64 * 64, activation=torch.nn.ReLU()),
    LinearModule(64 * 64, 64 * 64, activation=torch.nn.ReLU()),
    LinearModule(64 * 64, 10)
)

opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
loss = torch.nn.CrossEntropyLoss()
scorer = MulticlassScorer(".\\result\\classifier", 10)

core2 = Core(".\\result\\classifier", model, opt, loss, scorer)

model = torch.nn.Sequential(
    ConvModule(1, 64, 2, pool=torch.nn.MaxPool2d(2), activation=torch.nn.ReLU()),
    ConvModule(1, 64, 2, pool=torch.nn.MaxPool2d(2), activation=torch.nn.ReLU()),
    ConvModule(1, 64, 2, pool=torch.nn.MaxPool2d(2), activation=torch.nn.ReLU()),
    ConvModule(1, 64, 2, pool=torch.nn.MaxPool2d(2), activation=torch.nn.ReLU()),
    ConvModule(128, 256, 1, activation=torch.nn.ReLU()),
    torch.nn.Flatten(),
    LinearModule(256, 10)
)

opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
loss = torch.nn.CrossEntropyLoss()
scorer = MulticlassScorer(".\\result\\classifier", 10)

core3 = Core(".\\result\\classifier", model, opt, loss, scorer)

ensemble_core = VoteEnsemble(".\\result\\ensemble", [core, core2, core3], MulticlassScorer(".\\result\\classifier", 10))

ensemble_core.train(mnist_train, mnist_test, num_epochs=5)
ensemble_core.test(mnist_test)
