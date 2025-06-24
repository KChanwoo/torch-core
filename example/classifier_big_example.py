import os

import timm

from module.common import LinearModule
from module.vision import ConvModule

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch.nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from torchc.core import Core
from torchc.score import MulticlassScorer

# ImageNet dataset 경로 (실제 다운로드 및 경로 설정 필요)
imagenet_root = '/Volumes/Application/data'  # 실제 경로로 변경

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)

model = timm.create_model(
    'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
    pretrained=True,
    num_classes=100
)

opt = torch.optim.Adam(model.parameters(), lr=1.0e-2)
loss = torch.nn.CrossEntropyLoss()
scorer = MulticlassScorer("./result/classifier2")
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt,
    max_lr=1e-3,
    steps_per_epoch=1,
    epochs=5
)

core = Core("./result/classifier2", model, opt, loss, scheduler, scorer)

core.train(train_dataset, test_dataset, num_epochs=5, num_workers=4, batch_size=16)
# core.test(test_dataset, num_workers=0)
