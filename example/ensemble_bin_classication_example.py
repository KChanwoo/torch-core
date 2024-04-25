import os

from lib.ensemble import VoteEnsemble
from module.common import LinearModule
from module.vision import ConvModule

import torch.nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from lib.core import Core
from lib.score import MulticlassScorer, BinaryScorer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

mnist_train = list(filter(lambda i: i[1] == 0 or i[1] == 1, mnist_train))
mnist_test = list(filter(lambda i: i[1] == 0 or i[1] == 1, mnist_test))

mnist_train = [(item[0], [float(item[1])]) for item in mnist_train]
mnist_test = [(item[0], [float(item[1])]) for item in mnist_test]

# model 1
model = torch.nn.Sequential(
    ConvModule(1, 64, 2, pool=torch.nn.MaxPool2d(4), activation=torch.nn.ReLU()),
    ConvModule(64, 128, 2, pool=torch.nn.MaxPool2d(4), activation=torch.nn.ReLU()),
    ConvModule(128, 256, 1, activation=torch.nn.ReLU()),
    torch.nn.Flatten(),
    LinearModule(256, 1)
)

opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
loss = torch.nn.BCEWithLogitsLoss()
scorer = BinaryScorer("./result/classifier")

core = Core("./result/classifier", model, opt, loss, None, scorer)

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    LinearModule(28 * 28, 64 * 64, activation=torch.nn.ReLU()),
    LinearModule(64 * 64, 64 * 64, activation=torch.nn.ReLU()),
    LinearModule(64 * 64, 64 * 64, activation=torch.nn.ReLU()),
    LinearModule(64 * 64, 64 * 64, activation=torch.nn.ReLU()),
    LinearModule(64 * 64, 1)
)

opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
loss = torch.nn.BCEWithLogitsLoss()
scorer = BinaryScorer("./result/classifier2")

core2 = Core("./result/classifier2", model, opt, loss, None, scorer)

model = torch.nn.Sequential(
    ConvModule(1, 64, 3, 2, pool=torch.nn.MaxPool2d(2), activation=torch.nn.ReLU()),
    ConvModule(64, 128, 3, 2, pool=torch.nn.MaxPool2d(2), activation=torch.nn.ReLU()),
    ConvModule(128, 256, 1, activation=torch.nn.ReLU()),
    torch.nn.Flatten(),
    LinearModule(256, 1)
)

opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
loss = torch.nn.BCEWithLogitsLoss()
scorer = BinaryScorer("./result/classifier3")

core3 = Core("./result/classifier3", model, opt, loss, None, scorer)

ensemble_core = VoteEnsemble("./result/ensemble",
                             [core, core2, core3],
                             BinaryScorer("./result/ensemble"), mode=VoteEnsemble.SOFT,
                             weight=[.78, .87, .81])

# ensemble_core.train(mnist_train, mnist_test, num_epochs=5)
ensemble_core.test(mnist_test, test_all=False)
