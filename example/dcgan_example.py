import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from lib.core import Core, GanCore

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib

matplotlib.use('Agg')

import torch


class ChurchDataset(Dataset):
    def __init__(self, image_list: list[str]):
        self.image_list = image_list

    def __getitem__(self, index) -> T_co:
        image = self.image_list[index]

        im = Image.open(image)
        im = im.resize((64, 64))
        im = torch.tensor(np.array(im), dtype=torch.float32)
        return im.permute(2, 1, 0), [1]

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
        b, hw = x.shape
        return x.view((b,) + self.shape)


latent_size = (100,)

gen = torch.nn.Sequential(
    torch.nn.Linear(latent_size[0], 256 * 8 * 8),
    torch.nn.LeakyReLU(.2),
    torch.nn.BatchNorm1d(256 * 8 * 8),
    Reshape((256, 8, 8)),
    torch.nn.Upsample(scale_factor=2),
    torch.nn.Conv2d(256, 128, 5, padding='same'),
    torch.nn.LeakyReLU(.2),
    torch.nn.BatchNorm2d(128),
    torch.nn.Upsample(scale_factor=2),
    torch.nn.Conv2d(128, 64, 5, padding='same'),
    torch.nn.LeakyReLU(.2),
    torch.nn.BatchNorm2d(64),
    torch.nn.Upsample(scale_factor=2),
    torch.nn.Conv2d(64, 3, 5, padding='same'),
    torch.nn.Tanh()
)

gen_optim = torch.optim.Adam(params=gen.parameters(), lr=2e-4, weight_decay=8e-9)
gen_loss = torch.nn.BCELoss()

dis = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, 5, padding='same'),
    torch.nn.MaxPool2d(2),
    torch.nn.LeakyReLU(.2),
    torch.nn.Dropout(.3),
    torch.nn.BatchNorm2d(64),
    torch.nn.Conv2d(64, 128, 5, padding='same'),
    torch.nn.MaxPool2d(2),
    torch.nn.LeakyReLU(.2),
    torch.nn.Dropout(.3),
    torch.nn.BatchNorm2d(128),
    torch.nn.Flatten(),
    torch.nn.Linear(128 * 16 * 16, 1),
    torch.nn.Sigmoid()
)

dis_optim = torch.optim.Adam(params=dis.parameters(), lr=2e-4, weight_decay=8e-9)
dis_loss = torch.nn.BCELoss()

gen_core = Core(".\\result\\dcgan", gen, gen_optim, gen_loss)
dis_core = Core(".\\result\\dcgan", dis, dis_optim, dis_loss)

gan_optim = torch.optim.Adam(params=dis.parameters(), lr=2e-4, weight_decay=8e-9)
gan_loss = torch.nn.BCELoss()

gan = GanCore(".\\result\\dcgan", gen_core, dis_core, gan_optim, gan_loss, latent_size)

gan.train(train_dataset, num_epochs=40000, batch_size=64)
