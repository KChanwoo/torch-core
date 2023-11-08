import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms

from lib.core import Core, GanCore

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib

matplotlib.use('Agg')

import torch


class ChurchDataset(Dataset):
    def __init__(self, image_list: list[str]):
        self.image_list = image_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

    def __getitem__(self, index) -> T_co:
        image = self.image_list[index]

        im = Image.open(image).convert("RGB")
        im = im.resize((64, 64))
        return self.transform(im), [1]

    def __len__(self):
        return len(self.image_list)


image_dir = "F:\\data\\church_outdoor_train_lmdb\\expanded"
list_image = os.listdir(image_dir)

train_dataset = ChurchDataset([os.path.join(image_dir, item) for item in list_image])


class Reshape(torch.nn.Module):
    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        b = x.shape[0]
        return x.view((b,) + self.shape)


latent_size = (100,)

gen = torch.nn.Sequential(
    Reshape((100, 1, 1)),
    torch.nn.ConvTranspose2d(100, 64 * 8, 4, bias=False),
    torch.nn.BatchNorm2d(64 * 8),
    torch.nn.ReLU(True),
    torch.nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
    torch.nn.BatchNorm2d(64 * 4),
    torch.nn.ReLU(True),
    torch.nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
    torch.nn.BatchNorm2d(64 * 2),
    torch.nn.ReLU(True),
    torch.nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
    torch.nn.BatchNorm2d(64),
    torch.nn.ReLU(True),
    torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
    torch.nn.Tanh()
)

gen_optim = torch.optim.Adam(params=gen.parameters(), lr=2e-4, weight_decay=8e-9)
gen_loss = torch.nn.BCELoss()

dis = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, 4, 2, 1, bias=False),
    torch.nn.LeakyReLU(.2),
    torch.nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
    torch.nn.LeakyReLU(.2),
    torch.nn.BatchNorm2d(64 * 2),
    torch.nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
    torch.nn.LeakyReLU(.2),
    torch.nn.BatchNorm2d(64 * 4),
    torch.nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
    torch.nn.LeakyReLU(.2),
    torch.nn.BatchNorm2d(64 * 8),
    torch.nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
    Reshape((1,)),
    torch.nn.Sigmoid()
)

dis_optim = torch.optim.Adam(params=dis.parameters(), lr=2e-4, weight_decay=8e-9)
dis_loss = torch.nn.BCELoss()

gen_core = Core(".\\result\\dcgan", gen, gen_optim, gen_loss)
dis_core = Core(".\\result\\dcgan", dis, dis_optim, dis_loss)

gan = GanCore(".\\result\\dcgan", gen_core, dis_core, seed_shape=latent_size)

gan.train(train_dataset, num_epochs=40000, batch_size=64)
