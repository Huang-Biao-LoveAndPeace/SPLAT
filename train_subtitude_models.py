import argparse
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from networks.googlenet import googlenet
from networks.inception import inception_v3
from networks.resnet import resnet18
from networks.vgg import vgg13_bn
from networks.densenet import densenet121
from networks.mobilenetv2 import mobilenet_v2
from datasets import load_train_loader, load_valid_loader

parser = argparse.ArgumentParser(description='Proper ResNets for CIFAR in pytorch')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=101, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=50)
parser.add_argument('--device', default='cuda:0', type=str, help='cuda device')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100', 'tinyimagenet', 'tinyimagenetsub'],
                    help='dataset to use: cifar10, cifar100, or tinyimagenet')

def save_checkpoint(network, parameters, storedir, epoch=-1):
    if not os.path.exists(storedir):
        os.makedirs(storedir)

    if epoch == 0:
        path = os.path.join(storedir, 'untrained')
        params_path = os.path.join(storedir, 'parameters_untrained')
    else:
        path = os.path.join(storedir, 'last')
        params_path = os.path.join(storedir, 'parameters_last')

    torch.save(network.state_dict(), path)

    if parameters is not None:
        with open(params_path, 'wb') as outfile:
            pickle.dump(parameters, outfile, pickle.HIGHEST_PROTOCOL)
    print(f'[SAVE] The model was saved to {storedir}')

def main():
    global args, best_prec1, device
    args = parser.parse_args()
    device = args.device

    # Load data based on the selected dataset
    if args.dataset == 'cifar10':
        num_classes = 10
        train_loader = load_train_loader('cifar10', nbatch=args.batch_size)
        val_loader = load_valid_loader('cifar10', nbatch=args.batch_size)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_loader = load_train_loader('cifar100', nbatch=args.batch_size)
        val_loader = load_valid_loader('cifar100', nbatch=args.batch_size)
    elif args.dataset == 'tinyimagenet':
        num_classes = 200
        train_loader = load_train_loader('tinyimagenet', nbatch=args.batch_size)
        val_loader = load_valid_loader('tinyimagenet', nbatch=args.batch_size)
    elif args.dataset == 'tinyimagenetsub':
        num_classes = 10
        train_loader = load_train_loader('tinyimagenetsub', nbatch=args.batch_size)
        val_loader = load_valid_loader('tinyimagenetsub', nbatch=args.batch_size)

    model_names = ['densenet121', 'inception_v3', 'resnet18', 'vgg13_bn']
    model_list = [densenet121, inception_v3, resnet18, vgg13_bn]

    for model_name, tar_model in zip(model_names,model_list):
        # Set save directory based on the dataset
        save_dir = f"models/{args.dataset}/{args.dataset}_{model_name}"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_prec1 = 0
        model = tar_model(num_classes=num_classes)
        model.to(device)

        if args.resume:
            if os.path.isfile(args.resume):
                print(f"=> loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                print(f"=> no checkpoint found at '{args.resume}'")

        cudnn.benchmark = True

        # Define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        if args.half:
            model.half()
            criterion.half()

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 150], last_epoch=args.start_epoch - 1)

        if args.evaluate:
            validate(val_loader, model, criterion)
            return

        for epoch in range(args.start_epoch, args.epochs):
            print(f'current lr {optimizer.param_groups[0]["lr"]:.5e}')
            train(train_loader, model, criterion, optimizer, epoch)
            lr_scheduler.step()

            prec1 = validate(val_loader, model, criterion)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            # Use the new save_checkpoint function
            parameters = {'epoch': epoch + 1, 'best_prec1': best_prec1}
            save_checkpoint(model, parameters, save_dir, epoch)

        del model

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.to(device)
        input_var = input.to(device)
        target_var = target
        if args.half:
            input_var = input_var.half()

        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target

            if args.half:
                input_var = input_var.half()

            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    print(f' * Prec@1 {top1.avg:.3f}')
    return top1.avg

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()