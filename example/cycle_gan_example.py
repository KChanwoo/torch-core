import os
from collections import OrderedDict
from typing import Union

import numpy as np
from PIL import Image
from matplotlib import pyplot as plot
from torch import nn

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms
from tqdm import tqdm

from lib.core import Core, GanCore

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib

# matplotlib.use('Agg')

import torch


class HorseDataset(Dataset):
    def __init__(self, image_list: list[str], label_list: list[str]):
        self.image_list = image_list
        self.label_list = label_list
        self.transforms = transforms.Compose([
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

    def __getitem__(self, index) -> T_co:
        image = self.image_list[index]
        label = self.label_list[index]

        im = Image.open(image)
        im = np.array(im)
        im1 = self.transforms(torch.tensor(im, dtype=torch.float32).permute(2, 0, 1))
        im2 = Image.open(label)
        im2 = np.array(im2)
        im2 = self.transforms(torch.tensor(im2, dtype=torch.float32).permute(2, 0, 1))

        return im1, im2

    def __len__(self):
        return len(self.image_list)


image_dir1 = "F:\\data\\horse2zebra\\trainA"
image_dir2 = "F:\\data\\horse2zebra\\trainB"

image_dir1_test = "F:\\data\\horse2zebra\\testA"
image_dir2_test = "F:\\data\\horse2zebra\\testB"
list_image1 = os.listdir(image_dir1)
list_image2 = os.listdir(image_dir2)
list_image_test1 = os.listdir(image_dir1_test)
list_image_test2 = os.listdir(image_dir2_test)

train_dataset = HorseDataset([os.path.join(image_dir1, item) for item in list_image1], [os.path.join(image_dir2, item) for item in list_image2])
test_dataset = HorseDataset([os.path.join(image_dir1_test, item) for item in list_image_test1], [os.path.join(image_dir2_test, item) for item in list_image_test2])


latent_size = (100,)


class ResidualBlock(nn.Module):
    """
    residual block (He et al., 2016)
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        - Args
            in_channels: number of channels for an input feature map
            out_channels: number of channels for an output feature map

        - Note
            fixed a kernel_size to 3
        """
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.InstanceNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding='same', padding_mode='reflect'),
            nn.InstanceNorm2d(self.out_channels)
        )

    def forward(self, x):
        output = self.block(x) + x  # skip-connection

        return output


class Generator(nn.Module):
    def __init__(self, init_channel: int, kernel_size: int, stride: int, n_blocks: int = 6):
        """
        - Args
            stride: 2
            f_stride: 1/2
            kernel size: 9 (first and last), 3 for the others

            3 convolutions & 6 residual_blocks, 2 fractionally-strided convolutions
            one convolutions (features to RGB) -> 3 channel로 보낸다.

            instance normalization -> non-residual convolutional layers

            non-residual convolutional layers: followed by spatial batch normalization
            relu nonlinearities with the exception of the output layer
            + a scaled tanh to ensure that the output image has pixels in the range [0,255]
        """
        super(Generator, self).__init__()

        self.init_channel = init_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_blocks = n_blocks

        layers = OrderedDict()
        layers['conv_first'] = self._make_block(in_channels=3, out_channels=self.init_channel, kernel_size=7, stride=1,
                                                padding='same')  # first layer

        # downsampling path (d_k) -> two downsampling blocks
        for i in range(2):
            ic = self.init_channel * (i + 1)
            k = 2 * ic
            layers[f'd_{k}'] = self._make_block(in_channels=ic, out_channels=k, kernel_size=self.kernel_size,
                                                stride=self.stride)

        # residual block (R_k) -> 6 or 9 blocks
        for i in range(self.n_blocks):
            layers[f'R{k}_{i + 1}'] = ResidualBlock(k, k)  # in_channel = out_channel로 동일한 channel dimension 유지

        # upsampling path (u_k) -> two upsampling blocks
        for i in range(2):
            k = int(k / 2)
            layers[f'u_{k}'] = self._make_block(in_channels=k * 2, out_channels=k, kernel_size=self.kernel_size,
                                                stride=self.stride, mode='u')

        # last conv layer
        layers['conv_last'] = nn.Conv2d(in_channels=self.init_channel, out_channels=3, kernel_size=7, stride=1,
                                        padding='same', padding_mode='reflect')  # last conv layer (to rgb)
        layers['tanh'] = nn.Tanh()

        self.model = nn.Sequential(
            layers
        )

    def forward(self, x):
        op = self.model(x)
        assert op.shape == x.shape, f"output shape ({op.shape}) must be same with the input size ({x.shape})"
        return op

    def _make_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                    padding: Union[int, str] = 1, mode: str = 'd'):
        """
        builds a conv block

        - Args
            in_channels (int): # of channels of input feature map
            out_channels (int): # of channels of output feature map
            kernel_size (int): kernel size for a convolutional layer
            stride (int): stride for a convolution
            padding (int): an amount of padding for input feature map
            mode (str): 'd'(downsampling mode) or 'u'(upsampling mode) (default: 'd')
        """

        block = []
        if mode.lower() == 'd':
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   padding_mode='reflect'))

        elif mode.lower() == 'u':
            block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                            output_padding=1))  # output size를 input이랑 같게 해주려면 이렇게 설정을 해줄 수밖에 없음.

        block += [nn.InstanceNorm2d(out_channels), nn.ReLU(inplace=True)]

        return nn.Sequential(*block)


class Discriminator(nn.Module):
    def __init__(self, n_layers: int = 4, input_c: int = 3, n_filter: int = 64, kernel_size: int = 4):
        """
        - Args
            n_layers (int): number of convolutional layers in the network (default=3)
            input_c (int): number of input channels (default=3)
            n_filter (int): number of filters in the first convolutional layer (default=64)
            kernel_size (int): kernel size for every convolutional layers in the network (default=4)

        - Output
            2-D tensor (b,)

        PatchGAN 구조 사용 -> 이미지를 patch로 넣어주겠다는 것이 아님. receptive field를 이용하는 듯함.

        size of receptive fields = 1 + L(K-1); L = # of layers, K = kernel size (under the stride=1)
        이 receptive fields를 70으로 잡아주겠다.
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential()
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        layers = []

        # building conv block
        for i in range(self.n_layers):
            if i == 0:
                ic, oc = input_c, n_filter
                layers.append(
                    self._make_block(ic, oc, kernel_size=self.kernel_size, stride=2, padding=1, normalize=False))
            else:
                ic = oc
                oc = 2 * ic
                stride = 2

                if i == self.n_layers - 1:  # 마지막 레이어(c512)의 경우, stride=1로 설정할 것.
                    stride = 1

                layers.append(self._make_block(ic, oc, kernel_size=self.kernel_size, stride=stride, padding=1))

        # prediction
        layers.append(nn.Conv2d(oc, 1, kernel_size=self.kernel_size, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def _make_block(self, in_channels, out_channels, stride, kernel_size=3, padding=0, normalize=True):
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size,
                            padding=padding)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return nn.Sequential(*layers)


class CycleGanCore(GanCore):
    def __init__(self, base_path: str, model_g: Union[None, Core], model_d: Union[None, Core], test_dataset: Dataset):
        gan_optim = torch.optim.Adam(params=model_g.get_model().parameters(), lr=2e-4,
                                     betas=(.5, .999))
        gan_loss = torch.nn.L1Loss()
        super().__init__(base_path, model_g, model_d, optimizer=gan_optim, loss=gan_loss)
        self._test_dataset = test_dataset
        self._loader = None

    def train_step(self, model, inputs, labels, d_train=False):
        # the goal is make b using a
        a, b = inputs[:, 0, :, :], inputs[:, 1, :, :]
        # num_batch = a.shape[0]
        label_valid = torch.ones(1, device=self._device)

        with torch.set_grad_enabled(False):
            fake_b = self._model_g(a)

        if d_train:
            label_fake = torch.zeros(1, device=self._device)

            output_real, loss_real = self._model_d.train_step(self._model_d, [a, b], label_valid)
            output_fake, loss_fake = self._model_d.train_step(self._model_d, [a, fake_b.detach()], label_fake)

            full_loss = (loss_real + loss_fake) * .5
            output_, total_loss = output_fake, full_loss
        else:
            output_, loss_ = self._model_d.train_step(self._model_d, [a, fake_b], label_valid)  # MSE loss
            loss = self._loss(fake_b, b)  # L1 loss

            total_loss = loss_ + loss * 100

        return output_, total_loss

    def generate(self, number=1):
        self._loader = torch.utils.data.DataLoader(test_dataset, batch_size=number, num_workers=0, shuffle=True, collate_fn=self._default_collate)
        for item, label in tqdm(self._loader):
            a, b = item[:, 0, :, :], item[:, 1, :, :]
            b = b.to(self._device)
            return self._model_g(b)


gen = Generator()
dis = Discriminator()

gen_optim = torch.optim.Adam(params=gen.parameters(), lr=2e-4, weight_decay=8e-9)
gen_loss = torch.nn.MSELoss()

dis_optim = torch.optim.Adam(params=dis.parameters(), lr=2e-4, betas=(.5, .999))
dis_loss = torch.nn.MSELoss()

gen_core = Core(".\\result\\pixgan", gen, gen_optim, gen_loss)
dis_core = Core(".\\result\\pixgan", dis, dis_optim, dis_loss)

gan = CycleGanCore(".\\result\\pixgan", gen_core, dis_core, test_dataset=test_dataset)

gan.train(train_dataset, num_epochs=40000, batch_size=8)
