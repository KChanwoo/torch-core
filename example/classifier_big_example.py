import os

import torch.nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from torchc.core import Core
from torchc.score import MulticlassScorer
import timm

# ImageNet dataset 경로 (실제 다운로드 및 경로 설정 필요)
imagenet_root = '/Volumes/Application/data'  # 실제 경로로 변경

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)

model = timm.create_model('mobilevitv2_050.cvnets_in1k', pretrained=True, num_classes=100)

opt = torch.optim.Adam(model.parameters(), lr=1.0e-2)
loss = torch.nn.CrossEntropyLoss()
scorer = MulticlassScorer("./result/classifier2")

core = Core("./result/classifier2", model, opt, loss, None, scorer, early_stopping=True)

core.train(train_dataset, test_dataset, num_epochs=1000, num_workers=0)
# core.test(test_dataset, num_workers=0)
