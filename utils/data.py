import typing
from typing import List, Tuple
from torch._C import StringType
import torchvision
import torchvision.transforms as transforms
import torch

class Data:
    def __init__(self) -> None:
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=128,
                                                shuffle=True, num_workers=2)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=128,
                                                shuffle=False, num_workers=2)
        self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def get_classes(self) -> Tuple:
        return self.classes

    def get_trainloader(self) -> torch.utils.data.DataLoader:
        return self.trainloader

    def get_testloader(self) -> torch.utils.data.DataLoader:
        return self.testloader
