#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os
import pickle
import math
import time

#from apex import amp
from msnet_until.adaptive_inference import dynamic_evaluate
import networks
from datasets import load_train_loader, load_valid_loader, AverageMeter
from msnet_until.op_counter import measure_model
from msnet_until.ad import AD
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim

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


def arg_parser():
    parser = argparse.ArgumentParser(description='Image classification PK main script')
    model_names = ['msnet']

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
    arch_group.add_argument('--arch', '-a', metavar='ARCH', default='msnet', type=str, choices=model_names,
                            help='model architecture: ' + ' | '.join(model_names) + ' (default: msdnet)')
    arch_group.add_argument('--reduction', default=0.5, type=float, metavar='C', help='compression ratio of DenseNet'
                                                                                      ' (1 means dot\'t use compression) (default: 0.5)')
    arch_group.add_argument('--nBlocks', type=int, default=7)
    arch_group.add_argument('--nChannels', type=int, default=16)
    arch_group.add_argument('--base', type=int, default=4)
    arch_group.add_argument('--stepmode', type=str, choices=['even', 'lin_grow'], default='lin_grow')
    arch_group.add_argument('--step', type=str, default=2)
    arch_group.add_argument('--growthRate', type=int, default=6)
    arch_group.add_argument('--grFactor', default='1-2-4', type=str)
    arch_group.add_argument('--prune', default='max', choices=['min', 'max'])
    arch_group.add_argument('--bnFactor', default='1-2-4')
    arch_group.add_argument('--transFactor', default='2-5', type=str)
    arch_group.add_argument('--bottleneck', default=True, type=bool)
    arch_group.add_argument('--compress-factor', default=0.5, type=float)

    optim_group = parser.add_argument_group('optimization', 'optimization setting')
    optim_group.add_argument('--sgdr-t', default=10, type=int, dest='sgdr_t', help='SGDR T_0')
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
                             choices=['cosine', 'multistep', 'SGDR'])
    optim_group.add_argument('--decay-rate', default=0.1, type=float, metavar='N',
                             help='decay rate of learning rate (default: 0.1)')
    optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default=0.9)')
    optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                             help='weight decay (default: 1e-4)')

    return parser

def main():
    global args

    args = arg_parser().parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.grFactor = list(map(int, args.grFactor.split('-')))
    args.bnFactor = list(map(int, args.bnFactor.split('-')))
    args.transFactor = list(map(int, args.transFactor.split('-')))
    args.step = list(map(int, args.step.split('-')))
    args.nScales = len(args.grFactor)

    if args.data == 'cifar10' or args.data == 'tinyimagenetsub':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 200

    torch.manual_seed(args.seed)

    best_prec1, best_epoch = 0.0, 0

    global save_dir
    model_names = 'msnet'
    save_dir = os.path.join('models', args.data, f"{args.data}_{model_names}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.data.startswith('cifar'):
        IM_SIZE = 32
    else:
        IM_SIZE = 64

    # First Step: Initial Training
    model = getattr(networks, args.arch)(args)

    global n_flops
    n_flops, n_params = measure_model(model, IM_SIZE, IM_SIZE)
    del(model)

    model = getattr(networks, args.arch)(args).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
    else:
        model = torch.nn.DataParallel(model)

    cudnn.benchmark = True

    train_loader = load_train_loader(args.data, nbatch=args.batch_size)
    test_loader = load_valid_loader(args.data, nbatch=args.batch_size)
    val_loader = test_loader

    if args.evalmode is not None:
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

        if args.evalmode == 'anytime':
            validate(test_loader, model, criterion)
        else:
            dynamic_evaluate(model, test_loader, val_loader, args)
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
            print('Best var_prec1 {}'.format(best_prec1))

    print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))
    print('********** {} Final prediction results **********'.format(time.strftime('%m%d_%H:%M:%S')))
    validate(test_loader, model, criterion)

    # Save model state after the first training phase
    torch.save(model.state_dict(), os.path.join(save_dir, 'first_phase_model.pth'))

    # Second Step: Additional Training
    pred_model = getattr(networks, args.arch)(args).cuda()
    pred_model = torch.nn.DataParallel(pred_model)
    pred_model.load_state_dict(torch.load(os.path.join(save_dir, 'first_phase_model.pth')))
    print("=> loaded pretrained checkpoint for second phase")

    model = getattr(networks, args.arch)(args).cuda()
    data = torch.randn(2, 3, IM_SIZE, IM_SIZE).cuda()
    model.eval()
    with torch.no_grad():
        _, feat = model(data)

    trainable_list = nn.ModuleList([model])
    pad_reg = nn.ModuleList([AD(feat[j].size(1), feat[-1].size(1)).cuda() for j in range(args.nBlocks)])
    trainable_list.append(pad_reg)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(trainable_list.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
    else:
        model = torch.nn.DataParallel(model)

    cudnn.benchmark = True

    best_prec1, best_epoch = 0.0, 0

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_prec1, train_prec5, lr = train_second_phase(train_loader, model, pred_model, criterion, optimizer, epoch, pad_reg)
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion)

        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_prec1, val_prec1, train_prec5, val_prec5))

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            print('Best var_prec1 {}'.format(best_prec1))
            # Save model
            save_checkpoint(model, {'epoch': epoch, 'best_prec1': best_prec1}, save_dir, epoch)

    print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))
    print('********** {} Final prediction results **********'.format(time.strftime('%m%d_%H:%M:%S')))
    validate(test_loader, model, criterion)

    return

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()

    running_lr = None
    for i, (input, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)

        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        optimizer.zero_grad()
        output, _ = model(input_var)
        if not isinstance(output, list):
            output = [output]

        for j in range(len(output)):
            prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), input.size(0))
            top5[j].update(prec5.item(), input.size(0))

        criterion_div = DistillKL(4.0)
        loss = criterion(output[-1], target_var)

        #loss = 0.0
        for j in range(len(output) - 2, -1, -1):
            loss += 0.8 * criterion(output[j], target_var)
            loss += 0.9 * criterion_div(output[j], output[-1])

        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        #with amp.scale_loss(loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()
        optimizer.step()

        # measure elapsed time
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

def train_second_phase(train_loader, model, pred_model, criterion, optimizer, epoch, pad_reg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()
    pred_model.eval()

    end = time.time()

    running_lr = None
    for i, (input, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)

        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        optimizer.zero_grad()
        with torch.no_grad():
            _, pred_feat = pred_model(input_var)
        output, feat = model(input_var)
        if not isinstance(output, list):
            output = [output]

        for j in range(len(output)):
            prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), input.size(0))
            top5[j].update(prec5.item(), input.size(0))

        criterion_div = DistillKL(4.0)
        loss = criterion(output[-1], target_var)
        loss += pad_reg[-1](feat[-1], pred_feat[-1])

        # loss = 0.0
        for j in range(len(output) - 2, -1, -1):
            loss += 0.8 * criterion(output[j], target_var)
            loss += 0.9 * criterion_div(output[j], output[-1])
            loss += pad_reg[j](feat[j], pred_feat[-1])

        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()
        optimizer.step()

        # measure elapsed time
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

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep', eta_max=0.1, eta_min=0.):
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
    elif method == 'SGDR':
        i = int(math.log2(epoch / args.sgdr_t + 1))
        T_cur = epoch - args.sgdr_t * (2 ** (i) - 1)
        T_i = (args.sgdr_t * 2 ** i)

        lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * T_cur / T_i))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

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
            target = target.cuda(non_blocking=True)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            data_time.update(time.time() - end)

            output, _ = model(input_var)
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

            # measure elapsed time
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
    #for j in range(args.nBlocks):
        #print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    result_file = os.path.join(save_dir, 'AnytimeResults.txt')

    fd = open(result_file, 'w+')
    fd.write('AnytimeResults' + '\n')
    for j in range(args.nBlocks):
        test_str = (' @{ext}** flops {flops:.2f}M prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(ext=j + 1,flops=n_flops[j] / 1e6,top1=top1[j],top5=top5[j]))
        print(test_str)
        fd = open(result_file, 'a+')
        fd.write(test_str + '\n')
    fd.close()
    best_top1 = max([i.avg for i in top1])
    best_top5 = max([j.avg for j in top5])
    torch.save([e.avg for e in top1], os.path.join(save_dir, 'acc.pth'))

    return losses.avg, best_top1, best_top5

def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()