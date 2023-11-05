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
    torch.nn.Conv2d(1, 64, 2),
    torch.nn.MaxPool2d(4),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 128, 2),
    torch.nn.MaxPool2d(4),
    torch.nn.ReLU(),
    torch.nn.Conv2d(128, 256, 1),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(256, 10),
    torch.nn.Softmax()
)

opt = torch.optim.Adam(model.parameters(), lr=1.0e-4)
loss = torch.nn.CrossEntropyLoss()
scorer = MulticlassScorer(".\\result\\classifier", 10)

core = Core(".\\result\\classifier", model, opt, loss, scorer)

core.train(mnist_train, mnist_test, num_epochs=5)
core.test(mnist_test)
