from typing import Callable, Iterator, Tuple

import gpytorch
from vduq import vDUQ
import torch.optim as optim
import torch 
from utils.data import Data
import argparse

# This gives us hooks into torch training
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss, Metric
from ignite.contrib.handlers import ProgressBar

# This file contains the required code to launch the vDUQ model

def train(args : dict) -> None:
    data : Data = Data()
    trainloader : torch.utils.data.DataLoader = data.get_trainloader()
    testloader : torch.utils.data.DataLoader = data.get_testloader()
    classes : Tuple = data.get_classes()

    net = vDUQ(num_classes=len(classes),num_data=len(trainloader))

    optimizer : optim = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    dataiter : Iterator = iter(testloader)

    trainer = Engine(lambda _, data: vDUQ.training_step(data, optimizer, net))
    evaluator = Engine(lambda _, data: vDUQ.eval_step(data, net))


    metric = Average()
    metric.attach(trainer, "elbo")

    def output_transform(output):
        y_pred, y = output
        # Sample softmax values independently for classification at test time
        y_pred = y_pred.to_data_independent_dist()
        # The mean here is over likelihood samples
        y_pred = net.likelihood(y_pred).probs.mean(0)

        return y_pred, y

    metric : Metric = Accuracy(output_transform=output_transform)
    metric.attach(evaluator, "accuracy")

    metric : Metric= Loss(lambda y_pred, y: -net.elbo_fn(y_pred, y))
    metric.attach(evaluator, "elbo")

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(trainer)

    trainer.run(trainloader, max_epochs=args.epochs)
    pbar.attach(evaluator)

    evaluator.run(testloader)

    print(evaluator.state.metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train a vDUQ model')
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    train(args)
