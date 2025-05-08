import os
from typing import Union

import numpy as np
from PIL import Image
from matplotlib import pyplot as plot, pyplot as plt
from torch import nn, optim
from torch.nn import functional

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchvision import transforms
from tqdm import tqdm

from torchc.core import Core, GanCore

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib

matplotlib.use('Agg')

import torch


class CityDataset(Dataset):
    def __init__(self, image_list: list[str]):
        self.image_list = image_list
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

    def __getitem__(self, index) -> T_co:
        image = self.image_list[index]

        im = Image.open(image)
        im = np.array(im)
        im1 = self.transforms(im[:, :256])
        im2 = self.transforms(im[:, 256:])

        return im2, im1

    def __len__(self):
        return len(self.image_list)


image_dir = "F:\\data\\cityscapes\\train"
image_dir_test = "F:\\data\\cityscapes\\val"
list_image = os.listdir(image_dir)
list_image_test = os.listdir(image_dir_test)

train_dataset = CityDataset([os.path.join(image_dir, item) for item in list_image])
test_dataset = CityDataset([os.path.join(image_dir, item) for item in list_image_test])


latent_size = (100,)


# Conv -> Batchnorm -> Activate function Layer
'''
코드 단순화를 위한 convolution block 생성을 위한 함수
Encoder에서 사용될 예정
'''


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True, activation='relu'):
    layers = []

    # Conv layer
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))

    # Batch Normalization
    if bn:
        layers.append(nn.BatchNorm2d(c_out))

    # Activation
    if activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'none':
        pass

    return nn.Sequential(*layers)


# Deconv -> BatchNorm -> Activate function Layer
'''
코드 단순화를 위한 convolution block 생성을 위한 함수
Decoder에서 이미지 복원을 위해 사용될 예정
'''


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True, activation='lrelu'):
    layers = []

    # Deconv.
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))

    # Batchnorm
    if bn:
        layers.append(nn.BatchNorm2d(c_out))

    # Activation
    if activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'none':
        pass

    return nn.Sequential(*layers)


class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        # Unet encoder
        self.conv1 = conv(3, 64, 4, bn=False, activation='lrelu')  # (B, 64, 128, 128)
        self.conv2 = conv(64, 128, 4, activation='lrelu')  # (B, 128, 64, 64)
        self.conv3 = conv(128, 256, 4, activation='lrelu')  # (B, 256, 32, 32)
        self.conv4 = conv(256, 512, 4, activation='lrelu')  # (B, 512, 16, 16)
        self.conv5 = conv(512, 512, 4, activation='lrelu')  # (B, 512, 8, 8)
        self.conv6 = conv(512, 512, 4, activation='lrelu')  # (B, 512, 4, 4)
        self.conv7 = conv(512, 512, 4, activation='lrelu')  # (B, 512, 2, 2)
        self.conv8 = conv(512, 512, 4, bn=False, activation='relu')  # (B, 512, 1, 1)

        # Unet decoder
        self.deconv1 = deconv(512, 512, 4, activation='relu')  # (B, 512, 2, 2)
        self.deconv2 = deconv(1024, 512, 4, activation='relu')  # (B, 512, 4, 4)
        self.deconv3 = deconv(1024, 512, 4,
                              activation='relu')  # (B, 512, 8, 8) # Hint : U-Net에서는 Encoder에서 넘어온 Feature를 Concat합니다! (Channel이 2배)
        self.deconv4 = deconv(1024, 512, 4, activation='relu')  # (B, 512, 16, 16)
        self.deconv5 = deconv(1024, 256, 4, activation='relu')  # (B, 256, 32, 32)
        self.deconv6 = deconv(512, 128, 4, activation='relu')  # (B, 128, 64, 64)
        self.deconv7 = deconv(256, 64, 4, activation='relu')  # (B, 64, 128, 128)
        self.deconv8 = deconv(128, 3, 4, activation='tanh')  # (B, 3, 256, 256)

    # forward method
    def forward(self, input):
        # Unet encoder
        e1 = self.conv1(input)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)
        e8 = self.conv8(e7)

        # Unet decoder
        d1 = functional.dropout(self.deconv1(e8), 0.5, training=True)
        d2 = functional.dropout(self.deconv2(torch.cat([d1, e7], 1)), 0.5, training=True)
        d3 = functional.dropout(self.deconv3(torch.cat([d2, e6], 1)), 0.5, training=True)
        d4 = self.deconv4(torch.cat([d3, e5], 1))
        d5 = self.deconv5(torch.cat([d4, e4], 1))
        d6 = self.deconv6(torch.cat([d5, e3], 1))
        d7 = self.deconv7(torch.cat([d6, e2], 1))
        output = self.deconv8(torch.cat([d7, e1], 1))

        return output


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv2d(6, 64, 3, 2, 1),  # (3, 256, 256) to (64, 128, 128)
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(64, 128, 3, 2, 1),  # (128, 64, 64)
            torch.nn.BatchNorm2d(128, momentum=.8),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(128, 256, 3, 2, 1),  # (256, 32, 32)
            torch.nn.BatchNorm2d(256, momentum=.8),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(256, 512, 3, 2, 1),  # (512, 16, 16)
            torch.nn.BatchNorm2d(512, momentum=.8),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(512, 1, 3, 1, 1)  # (1, 16, 16)
        )

    def forward(self, x):
        a, b = x
        inputs = torch.concat([a, b], dim=1)

        return self.sequential(inputs)


class PixGanCore(GanCore):
    def __init__(self, base_path: str, model_g: Union[None, Core], model_d: Union[None, Core], test_dataset: Dataset):
        super().__init__(base_path, model_g, model_d, optimizer=None, loss=None, output_shape=(1, 16, 16))
        self._test_dataset = test_dataset
        self._loader = None

    def train_step(self, model, inputs, labels, d_train=False):
        # the goal is make b using a
        a, b = inputs, labels
        num_batch = a.shape[0]
        label_valid = self.create_real_label(num_batch)

        with torch.set_grad_enabled(False):
            fake_b = self._model_g(a)

        if d_train:
            label_fake = self.create_fake_label(num_batch)

            output_real, loss_real = self._model_d.train_step([a, b], label_valid, False)
            output_fake, loss_fake = self._model_d.train_step([a, fake_b.detach()], label_fake, False)

            output_, total_loss = output_fake, (loss_real + loss_fake) * .5
        else:

            output_, loss_ = self._model_d.train_step([a, fake_b], label_valid, False)  # MSE loss
            loss = self._model_g.get_criterion()(fake_b, b)  # L1 loss

            total_loss = loss_ + loss * 100

        return output_, total_loss

    def generate(self, number=1):
        self._loader = torch.utils.data.DataLoader(test_dataset, batch_size=number, num_workers=0, shuffle=False, collate_fn=self._default_collate)
        for item, label in tqdm(self._loader):
            a, b = item, label
            a = a.to(self._device)
            return self._model_g(a)


# gen = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=3, init_features=64, pretrained=False)
gen = Generator()
gen_optim = torch.optim.Adam(params=gen.parameters(), lr=2e-4, weight_decay=8e-9)
gen_loss = torch.nn.L1Loss()

dis = Discriminator()

dis_optim = torch.optim.Adam(params=dis.parameters(), lr=2e-4, betas=(.5, .999), weight_decay=1e-5)
dis_loss = torch.nn.MSELoss()

gen_core = Core(".\\result\\pixgan", gen, gen_optim, gen_loss)
dis_core = Core(".\\result\\pixgan", dis, dis_optim, dis_loss)

gan = PixGanCore(".\\result\\pixgan", gen_core, dis_core, test_dataset=test_dataset)

gan.train(train_dataset, num_epochs=40000, batch_size=8)



# # -1 ~ 1사이의 값을 0~1사이로 만들어준다
# def denorm(x):
#     out = (x + 1) / 2
#     return out.clamp(0, 1)
#
#
# # 이미지 시각화 함수
# def show_images(iter, real_a, real_b, fake_b):
#     plt.figure(figsize=(3, 1))
#     plt.subplot(131)
#     plt.imshow(real_a.cpu().data.numpy().transpose(1, 2, 0))
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.subplot(132)
#     plt.imshow(real_b.cpu().data.numpy().transpose(1, 2, 0))
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.subplot(133)
#     plt.imshow(fake_b.cpu().data.numpy().transpose(1, 2, 0))
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.savefig(os.path.join(".\\result\\pixgan", "{}.png".format(iter)))
#
#
# # Generator와 Discriminator를 GPU로 보내기
# G = Generator().cuda()
# D = Discriminator().cuda()
#
# criterionL1 = nn.L1Loss().cuda()
# criterionMSE = nn.MSELoss().cuda()
#
# # Setup optimizer
# g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
# d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
#
# train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=1, shuffle=True) # Shuffle
# test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=1, shuffle=False)
#
# # Train
# for epoch in range(1, 100):
#     for i, (real_a, real_b) in enumerate(train_loader, 1):
#         # forward
#         real_a, real_b = real_a.cuda(), real_b.cuda()
#         real_label = torch.ones(1).cuda()
#         fake_label = torch.zeros(1).cuda()
#
#         fake_b = G(real_a)  # G가 생성한 fake Segmentation mask
#
#         # ============= Train the discriminator =============#
#         # train with fake
#         fake_ab = torch.cat((real_a, fake_b), 1)
#         pred_fake = D.forward(fake_ab.detach())
#         loss_d_fake = criterionMSE(pred_fake, fake_label)
#
#         # train with real
#         real_ab = torch.cat((real_a, real_b), 1)
#         pred_real = D.forward(real_ab)
#         loss_d_real = criterionMSE(pred_real, real_label)
#
#         # Combined D loss
#         loss_d = (loss_d_fake + loss_d_real) * 0.5
#
#         # Backprop + Optimize
#         D.zero_grad()
#         loss_d.backward()
#         d_optimizer.step()
#
#         # =============== Train the generator ===============#
#         # First, G(A) should fake the discriminator
#         fake_ab = torch.cat((real_a, fake_b), 1)
#         pred_fake = D.forward(fake_ab)
#         loss_g_gan = criterionMSE(pred_fake, real_label)
#
#         # Second, G(A) = B
#         loss_g_l1 = criterionL1(fake_b, real_b) * 10
#
#         loss_g = loss_g_gan + loss_g_l1
#
#         # Backprop + Optimize
#         G.zero_grad()
#         D.zero_grad()
#         loss_g.backward()
#         g_optimizer.step()
#
#         if i % 200 == 0:
#             print(
#                 '======================================================================================================')
#             print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f'
#                   % (epoch, 100, i, len(train_loader), loss_d.item(), loss_g.item()))
#             print(
#                 '======================================================================================================')
#             show_images(str(epoch) + "_" + str(i), denorm(real_a.squeeze()), denorm(real_b.squeeze()), denorm(fake_b.squeeze()))