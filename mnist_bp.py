import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import network

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from util import accuracy, set_seed

class Opts:
    layer_size = 2000
    batch_size = 1000

    lr = 0.1
    weight_decay = 0
    epochs = 60
    
    seed = 0
    device = 'cuda'


@torch.no_grad()
def test(network_bp, test_loader, opts):
    all_outputs = []
    all_labels = []

    for (x_test, y_test) in test_loader:
        x_test, y_test = x_test.to(opts.device), y_test.to(opts.device)
        x_test = x_test.view(x_test.shape[0], -1)
        acts = network_bp(x_test)
        all_outputs.append(acts)
        all_labels.append(y_test)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    top1 = accuracy(all_outputs, all_labels, topk=(1,))[0]
    return top1

def train(network_bp, optimizer, train_loader, opts):
    running_loss = 0.

    for (x, y_ground) in train_loader:
        x, y_ground = x.to(opts.device), y_ground.to(opts.device)
        x = x.view(opts.batch_size, -1)

        with torch.enable_grad():
            ys = network_bp(x)
            loss = F.cross_entropy(ys, y_ground)
            loss.backward()

            running_loss += loss.detach()

        optimizer.step()
        optimizer.zero_grad()
        
    running_loss /= len(train_loader)
    return running_loss

def main(opts):
    set_seed(opts.seed)

    T_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    T_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(MNIST("~/data", train=True, download=True, transform=T_train), 
                              batch_size=opts.batch_size, shuffle=True, drop_last=True)
    
    test_loader = DataLoader(MNIST("~/data", train=False, download=True, transform=T_test), 
                             batch_size=opts.batch_size, shuffle=True)

    size = opts.layer_size
    network_bp = network.Network(dims=[28*28, size, size, size, 10], ff=False).to(opts.device)
    print(network_bp)

    optimizer = torch.optim.SGD(network_bp.parameters(), 
                                lr=opts.lr, 
                                weight_decay=opts.weight_decay)

    best_acc = 0.
    for step in range(1, opts.epochs+1):
        running_ce = train(network_bp, optimizer, train_loader, opts)

        top1 = test(network_bp, test_loader, opts)
        if top1 > best_acc:
            best_acc = top1
        print(f"Step {step:04d} CE: {running_ce:.4f} acc@1: {top1:.2f}")
    print('Best acc:', best_acc)

if __name__ == '__main__':
    opts = Opts()
    main(opts)