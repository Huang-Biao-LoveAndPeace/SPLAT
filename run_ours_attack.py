
"""
    Run attacks on the trained networks
"""
import os, json
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
# torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils

# custom libs
import models, utils
from datasets import NumpyDataset, TensorDataset, \
    load_train_loader, load_valid_loader

import attacks.l1 as l1
import attacks.l2 as l2
from attacks.PGDs import PGD, PGD_avg, PGD_max
from attacks.UAP import UAP
import attacks.ours_l1 as ours_l1
import attacks.ours_l2 as ours_l2
import attacks.ours_linf as ours_linf

from delay_attack_cost import compute_delay_metric_w_loader, get_rad_confidence_threshold


def get_ic_and_output(logits):
    for idx, ic_logit in enumerate(logits):
        p = nn.functional.softmax(ic_logit, dim=0)
        # 获取最大值及其对应的索引
        max_values, max_indices = torch.max(p, dim=0)
        return max_indices

# ------------------------------------------------------------------------------
#   Main attack code: Run or analysis
# ------------------------------------------------------------------------------
def run_attack(args, use_cuda=False):

    # load the clean validation set
    valid_loader = load_valid_loader(args.dataset, nbatch=args.batch_size)
    print ('[Run attack] load the valid set - {}'.format(args.dataset))

    # load the network
    netpath = os.path.join('models', '{}'.format(args.dataset))
    if 'sdn_ic_only' in args.nettype:
        netname = '{}_{}_{}'.format(args.dataset, args.network, args.nettype)
    else:
        netname = '{}_{}'.format(args.dataset, args.nettype)
    model, params = models.load_model(netpath, netname, epoch=-1)
    if use_cuda: model.cuda()
    model.eval()
    print ('[Run attack] load the model [{}], from [{}]'.format(netname, netpath))


    """
    导入替代模型的设置，是否集成，模型都加载在一个list中
    """
    sub_models = []
    if args.isensemble:
        model_names = ['densenet121', 'inception_v3', 'vgg13_bn']
    else:
        model_names = ['densenet121']
    for model_name in model_names:
        sub_net_name = '{}_{}'.format(args.dataset, model_name)
        sub_model, _ = models.load_model(netpath, sub_net_name, epoch=-1)
        sub_models.append(sub_model)

    save_folder = os.path.join( \
        'samples', args.dataset, netname)
    if not os.path.exists(save_folder): os.makedirs(save_folder)

    # run the DeepSloth + universal DeepSloth,
    #   and store them for the analysis
    if 'linf' == args.ellnorm:
        total_adv_data, total_adv_labels = \
            ours_linf.craft_per_sample_perturb_attack( \
                sub_models, valid_loader, max_iter=args.steps,per_iter=args.steps,epsilon=args.epsilion,eps_step=args.eps_step,device='cuda' if use_cuda else 'cpu')

    elif 'l2' == args.ellnorm:
        # > set the different parameters
        if 'tinyimagenet' in args.dataset:
            gamma = 0.05
        else:
            gamma = 0.1

        total_adv_data, total_adv_labels = \
            ours_l2.craft_per_sample_perturb_attack( \
                sub_models, valid_loader, max_iter=args.steps,per_iter=args.steps,gamma=gamma,
                device='cuda' if use_cuda else 'cpu')

    elif 'l1' == args.ellnorm:
        # > set the different parameters
        if 'tinyimagenet' in args.dataset:
            epsilon = 16; epsstep = 1.0
        else:
            epsilon =  8; epsstep = 0.5

        # > run
        total_adv_data, total_adv_labels = \
            ours_l1.craft_per_sample_perturb_attack( \
                sub_models, valid_loader, max_iter=args.steps,per_iter=args.steps,\
                epsilon=epsilon, eps_step=epsstep, \
                device='cuda' if use_cuda else 'cpu')

    else:
        assert False, ('Error: unsupported norm - {}'.format(args.ellnorm))

    """
        Take the max. iterations, since the attack is done
          with K (any number) iterations and save per K/10 iterations
    """

    """
    导入模型进行 query
    """
    threshold, _, _ = get_rad_confidence_threshold( \
        'models/{}', args.dataset, args.network, args.rad, device='cpu', \
        adv=False, cnnadv=False, sdnadv=False, netadv=False, nettype=args.nettype)

    model.confidence_threshold = threshold
    model.forward = model.early_exit
    model.output_to_return_when_ICs_are_delayed = 'network_output'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    real = total_adv_data[0]
    fake = total_adv_data[-1]

    ads_idx = []
    ads_output = []
    attack_data_iters = []
    attack_labels = []
    num = 0
    valid_loader = load_valid_loader(args.dataset, nbatch=1)

    for j, batch in enumerate(valid_loader):
        if j > 0 and j == len(real): break

        l = torch.from_numpy(np.array([real[j]])).to(device)
        r = torch.from_numpy(np.array([fake[j]])).to(device)

        output_logit1, e1, _ = model(l.to(device))
        output1 = get_ic_and_output(output_logit1)

        output_logit2, e2, _ = model(r.to(device))
        output2 = get_ic_and_output(output_logit2)

        if args.query_times == 0:
            print("unsported query times")
            break

        if output1 != output2:
            ads_idx.append(j)
            ads_output.append(output2)

            for i in range(args.query_times):
                output_logit1, e1, _ = model(l.to(device))
                output1 = get_ic_and_output(output_logit1)

                output_logit2, e2, _ = model(r.to(device))
                output2 = get_ic_and_output(output_logit2)

                b_x = 0.5 * l + 0.5 * r
                output_logit, e, _ = model(b_x.to(device))
                output = get_ic_and_output(output_logit)

                if output1 == output2:
                    break

                if output == output1:
                    l = b_x
                else:
                    r = b_x

        else:
            l = r

        b_x = l
        output_logit, mark, _ = model(b_x.to(device))
        output = get_ic_and_output(output_logit)
        # attack_data_iters.append((b_x.to(device)).cpu().detach().numpy()[0])
        attack_data_iters.append(b_x.to(device).cpu().detach().numpy()[0])
        attack_labels.append(batch[1].cpu())
        if output2 != output:
            num = num + 1

    attack_data_iters = np.asarray(attack_data_iters)

    with open(os.path.join(save_folder, '{}_{}_unqueried.pickle'.format(args.attacks, args.ellnorm)), 'wb') as handle:
        pickle.dump((total_adv_data[-1], total_adv_labels), handle, protocol=4)

    with open(os.path.join(save_folder, '{}_{}_queried.pickle'.format(args.attacks, args.ellnorm)), 'wb') as handle:
        pickle.dump((attack_data_iters, total_adv_labels), handle, protocol=4)

    # stop at here...
    exit()

def run_analysis(args, use_cuda=False):
    # load the clean validation set
    valid_loader = load_valid_loader(args.dataset)
    print ('[Run analysis] load the valid set - {}'.format(args.dataset))

    # load the network
    netpath = os.path.join('models', '{}'.format(args.dataset))
    if 'sdn_ic_only' in args.nettype:
        netname = '{}_{}_{}'.format(args.dataset, args.network, args.nettype)
    else:
        netname = '{}_{}'.format(args.dataset, args.nettype)
    model, params = models.load_model(netpath, netname, epoch='last')
    if use_cuda: model.cuda()
    model.eval()
    print ('[Run analysis] load the model [{}], from [{}]'.format(netname, netpath))

    save_folder = os.path.join( \
        'samples', args.dataset, netname)
    analyze_dir = os.path.join( \
        'analysis', args.dataset, netname)

    # create dir.
    if not os.path.exists(analyze_dir): os.makedirs(analyze_dir)
    print ('[Run analysis] create an analysis folder [{}]'.format(analyze_dir))

    # test configure
    datafiles = [
        os.path.join( \
            save_folder, \
            '{}_{}_{}.pickle'.format( \
                args.attacks, args.ellnorm, suffix))
        for suffix in ['unqueried', 'queried']
    ]
    rad_limits = [args.rad]

    # check the outputs
    for eachrad in rad_limits:
        for eachfile in datafiles:
            print ('--------')
            if 'univ' in eachfile:
                with open(eachfile, 'rb') as handle:
                    perturb = pickle.load(handle)
                    attack_data, attack_labels = ours_linf.apply_perturb_attack(valid_loader, perturb)
            else:
                with open(eachfile, 'rb') as handle:
                    attack_data, attack_labels = pickle.load(handle)

            # > save some samples
            samples_fname = os.path.join(analyze_dir, \
                '{}_samples.png'.format(eachfile.split(os.sep)[-1].replace('.pickle', '')))
            samples_size  = 8
            samples_data  = torch.from_numpy(attack_data[:samples_size])
            vutils.save_image(samples_data, samples_fname)

            # > compose dataset
            delayed_dataset= TensorDataset(attack_data, attack_labels)
            advdata_loader = DataLoader( \
                delayed_dataset, shuffle=False, batch_size=1)
            print(f'[{args.dataset}][{eachfile}] SDN evaluations')

            # > analyze
            analysis_file = os.path.join(analyze_dir, \
                '{}_{}_analysis'.format(eachfile.split(os.sep)[-1].replace('.pickle', ''), eachrad))
            plot_data, clean_auc, sloth_auc, clean_acc, sloth_acc = \
                compute_delay_metric_w_loader( \
                    'models/{}', args.dataset, args.network, \
                    eachrad, advdata_loader, analysis_file)
            print(f'[{args.dataset}][{eachfile}] RAD {eachrad}: Efficacy: {sloth_auc:.3f} - Accuracy: {sloth_acc:.3f}')

    print ('--------')
    print ('[Run analysis] Done.'); exit()




"""
    Main (Run the PGD/UAP/our attacks)
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser( \
        description='Run the our attacks.')

    # basic configurations
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='name of the dataset (cifar10, cifar100, tinyimagenetsub or tinyimagenet)')
    parser.add_argument('--network', type=str, default='vgg16bn',
                        help='location of the network (vgg16bn, resnet56, or mobilenet)')
    parser.add_argument('--nettype', type=str, default='sdn_ic_only',
                        help='location of the network (ex. msnet , msdnet , or sdn_ic_only / PGD_10_8_2_cnn --- for AT nets)')
    parser.add_argument('--rad', type=int, default=5,
                        help='early exit threshold of the network (ex. 5 or 15)')
    parser.add_argument('--runmode', type=str, default='attack',
                        help='runmode of the script (attack - crafts the adversarial samples, or analysis - computes the efficacy)')

    # attack configurations
    parser.add_argument('--attacks', type=str, default='ours',
                        help='the attack that this script will use (PGD, PGD-avg, PGD-max, UAP, ours)')
    parser.add_argument('--ellnorm', type=str, default='linf',
                        help='the norm used to bound the attack (default: linf - l1 and l2)')
    parser.add_argument('--isensemble', type=bool, default=False,
                        help='the number of samples consider (for UAP)')

    # hyper-parameters
    parser.add_argument('--steps', type=int, default=10,
                        help='the batch size used to craft adv. samples (default: 250)')
    parser.add_argument('--query_times', type=int, default=1,
                        help='the batch size used to craft adv. samples (default: 250)')
    parser.add_argument('--batch_size', type=int, default=250,
                        help='the batch size used to craft adv. samples (default: 250)')
    parser.add_argument('--epsilon', type=int, default=10,
                        help='the batch size used to craft adv. samples (default: 250)')
    parser.add_argument('--eps_step', type=int, default=10,
                        help='the batch size used to craft adv. samples (default: 250)')

    # execution parameters
    args = parser.parse_args()
    print (json.dumps(vars(args), indent=2))

    # set cuda if available
    set_cuda = True if 'cuda' == utils.available_device() else False
    print ('[{}] set cuda [{}]'.format(set_cuda, args.runmode))

    # run the attack or analysis
    if 'attack' == args.runmode:
        run_attack(args, use_cuda=set_cuda)
    elif 'analysis' == args.runmode:
        run_analysis(args, use_cuda=set_cuda)
    else:
        assert False, ('Error: undefined run-mode - {}'.format(args.runmode))
    # done.
