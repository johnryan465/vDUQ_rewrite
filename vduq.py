# This file contains the vDUQ model and training loop
# This relies on setting up a GP and a feature embedding and passing them to gp_dk
# In addition to a custom training loop

import torch
from torch import nn
import torch.nn.functional as F

from layers.inducing_gp import InducingGP
from layers.gp_dk import GP_DK
from layers.soft_spectral import soft_spectral_norm
import gpytorch
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO


class Embed(nn.Module):
    def __init__(self):
        super(Embed, self).__init__()
        self.conv1 = soft_spectral_norm(nn.Conv2d(3, 6, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = soft_spectral_norm(nn.Conv2d(6, 16, 5), coeff=0.9)
        self.fc1 = soft_spectral_norm(nn.Linear(16 * 5 * 5, 120), coeff=0.9)
        self.fc2 = soft_spectral_norm(nn.Linear(120, 84), coeff = 0.9)
        self.fc3 = soft_spectral_norm(nn.Linear(84, 10), coeff=0.9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class vDUQ(gpytorch.Module):
    def __init__(self, num_classes : int, num_data : int):
        super(vDUQ, self).__init__()
        self.gp = InducingGP(torch.ones(3,num_classes), num_classes=num_classes)
        self.embed = Embed()  # This is a regularised embedding layer with controlled lipshitz constant
        self.gp_dk = GP_DK(self.embed, self.gp)
        self.likelihood = SoftmaxLikelihood(
            num_classes=num_classes, mixing_weights=False)
        self.elbo_fn = VariationalELBO(self.likelihood, self.gp, num_data=num_data)


    def forward(self, x):
        return self.gp_dk.forward(x)

    @staticmethod
    def training_step(data, optimizer, net):
        net.train()
        net.likelihood.train()
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        elbo = -net.elbo_fn(outputs, labels)
        elbo.backward()
        optimizer.step()
        return elbo.item()

    @staticmethod
    def eval_step(data, net):
        net.eval()
        net.likelihood.eval()
        inputs, labels = data

        with torch.no_grad():
            outputs = net(inputs)

        return outputs, labels

