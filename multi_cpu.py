import argparse
parser = argparse.ArgumentParser(description='Train with multi-gpu')
parser.add_argument('-e', '--epochs', default=10, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=4096, type=int, metavar='N')
parser.add_argument('--lr', default=1e-2, type=float, metavar='N')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
parser.add_argument('-n', '--n-device', default=1, type=int, metavar='LR')
parser.add_argument('-d', '--dir', default='./data', type=str, metavar='DIR')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(map(str, range(args.n_device)))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '39392'

import sys
import time
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms, models
    

def main(rank, world_size, train_ds, val_ds, args):
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        pin_memory=True,
        sampler=DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False))

    print('==> Building model..')
    model = models.resnet152(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.lr, momentum=0.9)

    print('==> Training model')
    for epoch in range(args.epochs):
        train(ddp_model, train_loader, epoch, optimizer, criterion, rank)
        validate(ddp_model, val_loader, criterion, rank)


def train(model, train_loader, epoch, optimizer, criterion, device):
    print('Epoch: %d' % (epoch+1))

    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        disp_progress('Train', i, len(train_loader), train_loss, correct, total)
    print()


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            val_loss = loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            disp_progress('Validate', i, len(val_loader), val_loss, correct, total)
    print()


def disp_progress(mode, i, n, loss, correct, total):
    i += 1
    sys.stdout.write('\r%s: %d/%d==> Loss: %.6f | Acc: %.3f%% (%d/%d)'
        % (mode, i, n, loss/i, 100.*correct/total, correct, total))


if __name__ == '__main__':
    start = time.time()

    print('==> Preparing dataset..')
    image_size = 32
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_ds = datasets.CIFAR100(
        args.dir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]))

    val_ds = datasets.CIFAR100(
        args.dir,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    mp.spawn(main,
        args=(args.n_device, train_ds, val_ds, args),
        nprocs=args.n_device,
        join=True)

    end = time.time()
    print(f'Elapsed: {end-start:.2f}')

