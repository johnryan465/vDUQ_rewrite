from typing import Callable, Iterator, Tuple

import gpytorch
from vduq import vDUQ
import torch.optim as optim
import torch 
from utils.data import Data

# This file contains the required code to launch the vDUQ model

def train() -> None:
    data : Data = Data()
    trainloader : torch.utils.data.DataLoader = data.get_trainloader()
    testloader : torch.utils.data.DataLoader = data.get_testloader()
    classes : Tuple = data.get_classes()

    net = vDUQ(num_classes=len(classes),num_data=len(trainloader))

    optimizer : optim = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    training_loop: Callable[ [Tuple, torch.optim.Optimizer, gpytorch.Module], int] = net.training_loop()

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            running_loss += training_loop(data, optimizer, net)
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


    print('Finished Training')

    dataiter : Iterator = iter(testloader)

    images, labels = dataiter.next()
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

if __name__ == "__main__":
    train()
