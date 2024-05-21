import os

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

opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
loss = torch.nn.CrossEntropyLoss()
scorer = MulticlassScorer("./result/classifier")

core = Core("./result/classifier", model, opt, loss, None, scorer)

core.train(mnist_train, mnist_test, num_epochs=5)
core.test(mnist_test)
