"""
    Run attacks on the trained networks
"""
import os, json
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm

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
import attacks.deepsloth_l1 as ours_l1
import attacks.deepsloth_l2 as ours_l2
import attacks.deepsloth_linf as ours_linf

from delay_attack_cost import compute_delay_metric_w_loader

# ------------------------------------------------------------------------------
#   Main attack code: analysis
# ------------------------------------------------------------------------------


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
    if 'sdn_ic_only' in args.sub_nettype:
        sub_netname = '{}_{}_{}'.format(args.dataset, args.sub_network, args.sub_nettype)
    else:
        sub_netname = '{}_{}'.format(args.dataset, args.sub_nettype)
    model, params = models.load_model(netpath, netname, epoch='last')
    if use_cuda: model.cuda()
    model.eval()
    print ('[Run analysis] load the model [{}], from [{}]'.format(netname, netpath))

    """
        Perform analysis
    """

    if 'transfer' == args.attacks:
        save_folder = os.path.join( \
            'samples', args.dataset, sub_netname)
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
                    'deepsloth', args.ellnorm, suffix))
            for suffix in ['persample']
        ]
        rad_limits = [5, 15]

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
                    'transfer_{}_samples.png'.format(eachfile.split(os.sep)[-1].replace('.pickle', '')))
                samples_size  = 8
                samples_data  = torch.from_numpy(attack_data[:samples_size])
                vutils.save_image(samples_data, samples_fname)

                # > compose dataset
                delayed_dataset= TensorDataset(attack_data, attack_labels)
                advdata_loader = DataLoader( \
                    delayed_dataset, shuffle=False, batch_size=1)
                print(f'[{args.dataset}][{eachfile}] SDN evaluations')

                # > analyze
                analysis_file = os.path.join(analyze_dir, '{}_{}_{}_transfer_to_{}_analysis'.format(
                                                 eachfile.split(os.sep)[-1].replace('.pickle', ''), eachrad,
                                                 sub_netname, netname))
                plot_data, clean_auc, sloth_auc, clean_acc, sloth_acc = \
                    compute_delay_metric_w_loader( \
                        'models/{}', args.dataset, args.network, \
                        eachrad, advdata_loader, analysis_file)
                print(f'[{args.dataset}][{eachfile}] RAD {eachrad}: Efficacy: {sloth_auc:.3f} - Accuracy: {sloth_acc:.3f}')

        print ('--------')
        print ('[Run analysis] Done.'); exit()
        # stop here...

    elif 'transfer-class' == args.attacks:
        save_folder = os.path.join( \
            'samples', args.dataset, sub_netname)
        analyze_dir = os.path.join( \
            'analysis', args.dataset, netname)

        # create dir.
        if not os.path.exists(analyze_dir): os.makedirs(analyze_dir)
        print ('[Run analysis] create an analysis folder [{}]'.format(analyze_dir))

        # test configure
        tot_class = list(range(10))
        datafiles = [
            os.path.join( \
                save_folder, \
                '{}_{}_class_{}.pickle'.format( \
                    args.attacks, args.ellnorm, each_class))
            for each_class in tot_class
        ]
        rad_limits = [5, 15]

        # check the outputs
        for eachrad in rad_limits:
            print ('-------- [RAD < {}] --------'.format(eachrad))

            # > loop over the files
            tot_caccuracy, tot_cauc = 0., 0.
            tot_aaccuracy, tot_aauc = 0., 0.
            for eachfile in datafiles:
                # >> loader
                each_class  = int(eachfile.replace('.pickle', '').split('_')[-1])
                data_loader = utils.ManualData.get_loader(utils.ManualData( \
                    *utils.get_task_class_data(args.dataset, get_class=each_class)[2:]), batch_size=args.batch_size)

                # >> load the perturbation
                with open(eachfile, 'rb') as handle:
                    perturb = pickle.load(handle)
                attack_data, attack_labels = \
                    ours_linf.apply_perturb_attack(data_loader, perturb)

                # >> save some samples
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

                # >> analyze
                analysis_file = os.path.join(analyze_dir, \
                    '{}_{}_{}_transfer_to_{}_analysis'.format(eachfile.split(os.sep)[-1].replace('.pickle', ''), eachrad, sub_netname, netname))
                plot_data, clean_auc, sloth_auc, clean_acc, sloth_acc = \
                    compute_delay_metric_w_loader( \
                        'models/{}', args.dataset, args.network, \
                        eachrad, advdata_loader, analysis_file)
                print(f'[{args.dataset}][{eachfile}] RAD {eachrad}: Efficacy: {sloth_auc:.3f} - Accuracy: {sloth_acc:.3f}')

                # >> store
                tot_cauc      += clean_auc
                tot_caccuracy += clean_acc
                tot_aauc      += sloth_auc
                tot_aaccuracy += sloth_acc
            # > end for ...

            # > report the averages
            print ('[Run analysis] totals')
            print ('  [clean] efficacy: {:.4f} / accuracy: {:.4f} (avg.)'.format( \
                tot_cauc / len(tot_class), tot_caccuracy / len(tot_class)))
            print ('  [sloth] efficacy: {:.4f} / accuracy: {:.4f} (avg.)'.format( \
                tot_aauc / len(tot_class), tot_aaccuracy / len(tot_class)))

        print ('--------')
        print ('[test_deepsloth] done.'); exit()
        # stop here...

    else:
        assert False, ('Error: unsupported attacks - {}'.format(args.attacks))
    # done.


"""
    Main (Run our attacks)
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser( \
        description='Run the PGD/UAP/our attacks.')

    # basic configurations
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='name of the dataset (cifar10 or tinyimagenet)')
    parser.add_argument('--network', type=str, default='vgg16bn',
                        help='location of the network (vgg16bn, resnet56, or mobilenet)')
    parser.add_argument('--nettype', type=str, default='sdn_ic_only',
                        help='location of the network (ex. cnn, or sdn_ic_only / PGD_10_8_2_cnn --- for AT nets)')
    parser.add_argument('--runmode', type=str, default='analysis',
                        help='runmode of the script (attack - crafts the adversarial samples, or analysis - computes the efficacy)')

    # attack configurations
    parser.add_argument('--attacks', type=str, default='transfer',
                        help='the attack that this script will use (transfer, transfer-class)')
    parser.add_argument('--ellnorm', type=str, default='linf',
                        help='the norm used to bound the attack (default: linf - l1 and l2)')

    # hyper-parameters
    parser.add_argument('--batch-size', type=int, default=250,
                        help='the batch size used to craft adv. samples (default: 250)')

    parser.add_argument('--sub_network', type=str, default='vgg16bn',
                        help='location of the subtitude network (vgg16bn, resnet56, or mobilenet)')
    parser.add_argument('--sub_nettype', type=str, default='sdn_ic_only',
                        help='location of the subtitude network (ex. cnn, or sdn_ic_only / PGD_10_8_2_cnn --- for AT nets)')

    # execution parameters
    args = parser.parse_args()
    print (json.dumps(vars(args), indent=2))

    # set cuda if available
    set_cuda = True if 'cuda' == utils.available_device() else False
    print ('[{}] set cuda [{}]'.format(set_cuda, args.runmode))

    # run the attack or analysis
    if 'analysis' == args.runmode:
        run_analysis(args, use_cuda=set_cuda)
    else:
        assert False, ('Error: undefined run-mode - {}'.format(args.runmode))
    # done.
