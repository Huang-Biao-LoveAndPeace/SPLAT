import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim


import time
import utils
from tqdm import tqdm


def craft_per_sample_perturb_attack( \
    models, test_loader, \
    max_iter=30, per_iter=10, epsilon=0.03125, eps_step=0.03125, nbatch=10, device='cpu'):
    print (' [SPLAT - Linf] start crafting per-sample attacks...')
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models:
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.eval()
        model.to(device)

    # max iter and save parameters
    attack_data_iters = [list() for _ in range(int((max_iter/per_iter))+2)]
    attack_labels = []

    # to measure the time
    start_time = time.time()

    # loop over the test set
    for batch_idx, batch in enumerate(test_loader):
        # if nbatch > 0 and batch_idx == nbatch: break

        # : original dataset
        b_x = batch[0].to(device, dtype=torch.float)
        attack_data_iters[0].append(b_x.cpu().detach().numpy())  # unperturbed

        ori_b_x = b_x.data
        attack_data_iters[1].append(b_x.cpu().detach().numpy())  # with l_inf noise

        # : do perturbations
        for ii in tqdm(range(1, max_iter+1), desc='[SPLAT-{}]'.format(batch_idx)):
            b_x.requires_grad = True
            predicted = torch.zeros_like(models[0](b_x))
            for model in models:
                model.zero_grad()
                predicted = predicted + model(b_x)
            predicted = predicted / len(models)
            p = (torch.ones(predicted.shape) / predicted.shape[1]).to(predicted.device)
            cost = loss_fn(predicted, p)
            cost.backward(retain_graph=True)

            adv_cur_x = b_x - eps_step * b_x.grad.sign()
            perturb = torch.clamp(adv_cur_x - ori_b_x, min=-epsilon, max=epsilon)
            b_x = torch.clamp(ori_b_x + perturb, min=0, max=1).detach_()

            print(cost)
            del cost,predicted

            if ii % per_iter == 0:
                attack_data_iters[int(ii/per_iter)+1].append(b_x.cpu().detach().numpy())

        attack_labels.extend(batch[1])

    # to measure the time
    termi_time = time.time()
    print (' [SPLAT - Linf] time taken for crafting 10k samples: {:.4f}'.format(termi_time - start_time))

    # return the data
    attack_data_iters = [np.vstack(attack_data) for attack_data in attack_data_iters]
    attack_labels = np.asarray(attack_labels)
    return attack_data_iters, attack_labels
