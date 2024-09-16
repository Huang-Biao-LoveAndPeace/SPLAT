import argparse
import os
import time
import math
import pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from datasets import load_train_loader, load_valid_loader
import networks


def save_model(network, parameters, storedir, epoch=-1):
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


def arg_parser():
    parser = argparse.ArgumentParser(description='Image classification PK main script')
    model_names = ['msdnet']

    exp_group = parser.add_argument_group('exp', 'experiment setting')
    exp_group.add_argument('--resume', action='store_true', help='path to latest checkpoint (default: none)')
    exp_group.add_argument('--evalmode', default=None, choices=['anytime', 'dynamic'], help='which mode to evaluate')
    exp_group.add_argument('--evaluate-from', default=None, type=str, metavar='PATH',
                           help='path to saved checkpoint (default: none)')
    exp_group.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                           help='print frequency (default: 100)')
    exp_group.add_argument('--seed', default=0, type=int, help='random seed')
    exp_group.add_argument('--gpu', default=None, type=str, help='GPU available.')

    data_group = parser.add_argument_group('data', 'dataset setting')
    data_group.add_argument('--data', metavar='D', default='cifar10',
                            choices=['cifar10', 'cifar100', 'tinyimagenet'], help='data to work on')
    data_group.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')

    arch_group = parser.add_argument_group('arch', 'model architecture setting')
    arch_group.add_argument('--arch', '-a', metavar='ARCH', default='msdnet', type=str, choices=model_names,
                            help='model architecture: ' + ' | '.join(model_names) + ' (default: msdnet)')
    arch_group.add_argument('--reduction', default=0.5, type=float, metavar='C', help='compression ratio of DenseNet'
                                                                                      ' (1 means dot\'t use compression) (default: 0.5)')
    arch_group.add_argument('--nBlocks', type=int, default=7)
    arch_group.add_argument('--nChannels', type=int, default=16)
    arch_group.add_argument('--base', type=int, default=4)
    arch_group.add_argument('--stepmode', type=str, choices=['even', 'lin_grow'], default='lin_grow')
    arch_group.add_argument('--step', type=int, default=2)
    arch_group.add_argument('--growthRate', type=int, default=6)
    arch_group.add_argument('--grFactor', default='1-2-4', type=str)
    arch_group.add_argument('--prune', default='max', choices=['min', 'max'])
    arch_group.add_argument('--bnFactor', default='1-2-4')
    arch_group.add_argument('--bottleneck', default=True, type=bool)

    optim_group = parser.add_argument_group('optimization', 'optimization setting')
    optim_group.add_argument('--epochs', default=201, type=int, metavar='N',
                             help='number of total epochs to run (default: 164)')
    optim_group.add_argument('--start-epoch', default=0, type=int, metavar='N',
                             help='manual epoch number (useful on restarts)')
    optim_group.add_argument('-b', '--batch_size', default=125, type=int, metavar='N',
                             help='mini-batch size (default: 64)')
    optim_group.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                             help='initial learning rate (default: 0.1)')
    optim_group.add_argument('--lr-type', default='multistep', type=str, metavar='T',
                             help='learning rate strategy (default: multistep)',
                             choices=['cosine', 'multistep'])
    optim_group.add_argument('--decay-rate', default=0.1, type=float, metavar='N',
                             help='decay rate of learning rate (default: 0.1)')
    optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default=0.9)')
    optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                             help='weight decay (default: 1e-4)')

    return parser


def main():
    global args
    args = arg_parser().parse_args()

    model_names = 'msdnet'

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.nScales = len(args.grFactor)

    if args.data == 'cifar10' or args.data == 'tinyimagenetsub':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 200

    torch.manual_seed(args.seed)

    best_prec1, best_epoch = 0.0, 0

    save_dir = os.path.join('models', args.data, f"{args.data}_{model_names}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.data.startswith('cifar'):
        IM_SIZE = 32
    else:
        IM_SIZE = 64

    model = getattr(networks, args.arch)(args)
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    train_loader = load_train_loader(args.data, nbatch=args.batch_size)
    test_loader = load_valid_loader(args.data, nbatch=args.batch_size)
    val_loader = test_loader

    if args.evalmode is not None:
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

        if args.evalmode == 'anytime':
            validate(test_loader, model, criterion)
        return

    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1'
              '\tval_prec1\ttrain_prec5\tval_prec5']

    model.forward = model.train_forward

    for epoch in range(args.start_epoch, args.epochs):

        train_loss, train_prec1, train_prec5, lr = train(train_loader, model, criterion, optimizer, epoch)

        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion)

        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_prec1, val_prec1, train_prec5, val_prec5))

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            print('Best val_prec1 {}'.format(best_prec1))

            # Save model
            save_model(model,  {'epoch': epoch, 'best_prec1': best_prec1}, save_dir, epoch)

    print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))

    print('********** Final prediction results **********')
    validate(test_loader, model, criterion)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.train()

    end = time.time()

    running_lr = None
    for i, (input, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)

        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        target = target.cuda(device=None)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        if not isinstance(output, list):
            output = [output]

        loss = 0.0
        for j in range(len(output)):
            loss += criterion(output[j], target_var)

        losses.update(loss.item(), input.size(0))

        for j in range(len(output)):
            prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), input.size(0))
            top5[j].update(prec5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f}\t'
                  'Acc@1 {top1.val:.4f}\t'
                  'Acc@5 {top5.val:.4f}'.format(
                epoch, i + 1, len(train_loader),
                batch_time=batch_time, data_time=data_time,
                loss=losses, top1=top1[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                    i + 1, len(val_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top5=top5[-1]))
    for j in range(args.nBlocks):
        print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
    return losses.avg, top1[-1].avg, top5[-1].avg


def load_checkpoint(args):
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state


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
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data.startswith('cifar'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()