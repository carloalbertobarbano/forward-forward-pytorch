import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import network
import torch.utils.tensorboard

from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from util import set_seed, accuracy

class Opts:
    hard_negatives = True
    layer_size = 2000

    batch_size = 200
    lr = 0.0001
    weight_decay = 0
    epochs = 60
    steps_per_block = 60
    theta = 10.
    
    seed = 0
    device = 'cuda'


def norm_y(y_one_hot: torch.Tensor):
    return y_one_hot.sub(0.1307).div(0.3081)

@torch.no_grad()
def test(network_ff, linear_cf, test_loader, opts):
    all_outputs = []
    all_labels = []
    all_logits = []

    for (x_test, y_test) in test_loader:
        x_test, y_test = x_test.to(opts.device), y_test.to(opts.device)
        x_test = x_test.view(x_test.shape[0], -1)

        acts_for_labels = []

        # slow method
        for label in range(10):
            test_label = torch.ones_like(y_test).fill_(label)
            test_label = norm_y(F.one_hot(test_label, num_classes=10))
            x_with_labels = torch.cat((x_test, test_label), dim=1)
            
            acts = network_ff(x_with_labels)
            acts = acts.norm(dim=-1)
            acts_for_labels.append(acts)
        
        # these are logits
        acts_for_labels = torch.stack(acts_for_labels, dim=1) #should be BSZxLABELSxLAYERS (10)
        all_outputs.append(acts_for_labels)
        all_labels.append(y_test)

        # quick method
        neutral_label = norm_y(torch.full((x_test.shape[0], 10), 0.1, device=opts.device))
        acts = network_ff(torch.cat((x_test, neutral_label), dim=1))
        logits = linear_cf(acts.view(acts.shape[0], -1))
        all_logits.append(logits)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    slow_acc = accuracy(all_outputs.mean(dim=-1), all_labels, topk=(1,))[0]
    fast_acc = accuracy(all_logits, all_labels, topk=(1,))[0]
    return slow_acc, fast_acc

def train(network_ff, optimizer, linear_cf, optimizer_cf, train_loader, start_block, opts):
    running_loss = 0.
    running_ce = 0.

    for (x, y_pos) in train_loader:
        x, y_pos = x.to(opts.device), y_pos.to(opts.device)
        x = x.view(opts.batch_size, -1)

        # positive pairs
        y_pos_one_hot = norm_y(F.one_hot(y_pos, num_classes=10))
        x_pos = torch.cat((x, y_pos_one_hot), dim=1)
        
        # sample negatives (and train linear cf)
        with torch.no_grad():
            ys = network_ff(torch.cat((x, torch.ones_like(y_pos_one_hot).fill_(0.1)), dim=1))

        with torch.enable_grad():
            logits = linear_cf(ys.view(ys.shape[0], -1).detach())
            ce = F.cross_entropy(logits, y_pos)
            ce.backward()
            running_ce += ce.detach()

        optimizer_cf.step()
        optimizer_cf.zero_grad()

        # negative pairs from softmax layer
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        idx = torch.where(preds != y_pos)
        y_hard_one_hot = norm_y(F.one_hot(preds, num_classes=10))
        x_hard = torch.cat((x, y_hard_one_hot), dim=1)[idx]

        # negative pairs from random labels
        y_rand = torch.randint(0, 10, (opts.batch_size,), device=opts.device)
        idx = torch.where(y_rand != y_pos) # correct labels
        y_rand_one_hot = norm_y(F.one_hot(y_rand, num_classes=10))
        x_rand = torch.cat((x, y_rand_one_hot), dim=1) #[idx] # keeping positives seems to work better

        x_neg = x_rand
        if opts.hard_negatives:
            x_neg = torch.cat((x_neg, x_hard), dim=0)
            
        with torch.enable_grad():
            z_pos = network_ff(x_pos, cat=False)
            z_neg = network_ff(x_neg, cat=False)

            for idx, (zp, zn) in enumerate(zip(z_pos, z_neg)):
                if idx < start_block:
                    continue

                positive_loss = torch.log(1 + torch.exp((-zp.norm(dim=-1) + opts.theta))).mean()
                negative_loss = torch.log(1 + torch.exp((zn.norm(dim=-1) - opts.theta))).mean()
                loss = positive_loss + negative_loss
                loss.backward()

                running_loss += loss.detach()
                optimizer[idx].step()
                optimizer[idx].zero_grad()
    
    running_loss /= len(train_loader)
    running_ce /= len(train_loader)

    return running_loss, running_ce

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
                              batch_size=opts.batch_size, shuffle=True, drop_last=True, num_workers=8,
                              persistent_workers=True)
    
    test_loader = DataLoader(MNIST("~/data", train=False, download=True, transform=T_test), 
                             batch_size=opts.batch_size, shuffle=True, num_workers=8,
                             persistent_workers=True)

    size = opts.layer_size
    network_ff = network.Network(dims=[28*28 + 10, size, size, size, size]).to(opts.device)
    print(network_ff)

    # Create one optimizer for evey relu layer (block)
    optimizers = [
        torch.optim.Adam(block.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
            for block in network_ff.blocks.children()
    ] 

    # Softmax layer for predicting classes from embeddings (fast method)
    linear_cf = nn.Linear(size*network_ff.n_blocks, 10).to(opts.device)
    optimizer_cf = torch.optim.Adam(linear_cf.parameters(), lr=0.0001)

    writer = SummaryWriter()

    start_block = 0
    for step in range(1, opts.epochs+1):
        running_loss, running_ce = train(network_ff, optimizers, linear_cf, optimizer_cf,
                                         train_loader, start_block, opts)
        if step % opts.steps_per_block == 0:
            if start_block+1 < network_ff.n_blocks:
                start_block += 1
                print("Freezing block", start_block-1)

        writer.add_scalar("train/loss", running_loss, step)
        writer.add_scalar("train/ce", running_ce, step)

        train_slow_acc, train_fast_acc = test(network_ff, linear_cf, train_loader, opts)
        test_slow_acc, test_fast_acc = test(network_ff, linear_cf, test_loader, opts)

        writer.add_scalar("acc_fast/train", train_fast_acc, step)
        writer.add_scalar("acc_fast/test", test_fast_acc, step)
        writer.add_scalar("acc_slow/train", train_slow_acc, step)
        writer.add_scalar("acc_slow/test", test_slow_acc, step)

        print(f"Step {step:03d} Loss: {running_loss:.4f} CE: {running_ce:.4f}",
              f"-- TRAIN: fast {train_fast_acc:.2f} (err {(100. - train_fast_acc):.2f}) slow {train_slow_acc:.2f} (err {(100. - train_slow_acc):.2f})",
              f"-- TEST: fast {test_fast_acc:.2f} (err {(100. - test_fast_acc):.2f}) - slow {test_slow_acc:.2f} (err {(100. - test_slow_acc):.2f})")

if __name__ == '__main__':
    opts = Opts()
    main(opts)