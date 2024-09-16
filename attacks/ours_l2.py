import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim


import time
import utils
from tqdm import tqdm


def craft_per_sample_perturb_attack( \
    models, test_loader, \
    max_iter=550, per_iter=50, gamma=0.1, init_norm=1., nbatch=10, device='cpu'):
    print(' [SPLAT - L2] start crafting per-sample attacks...')
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models:
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.eval()
        model.to(device)

    # max iter and save parameters
    attack_data_iters = [list() for _ in range(int((max_iter/per_iter))+2)] # returns unperturbed, noisy and the perturbed for every per_iter iterations
    attack_labels = []

    # to measure the time
    start_time = time.time()

    # loop over the test set
    for batch_idx, batch in enumerate(test_loader):
        # if nbatch > 0 and batch_idx == nbatch: break

        attack_labels.extend(batch[1])

        # : original dataset
        b_x = batch[0].to(device, dtype=torch.float)
        if b_x.min() < 0 or b_x.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        attack_data_iters[0].append(b_x.cpu().detach().numpy())  # unperturbed

        r = np.random.randn(*b_x.shape)
        norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=-1).reshape(-1, 1, 1, 1)
        delta =  torch.zeros_like((r / norm) * init_norm).to(device).float() # init delta
        delta.requires_grad_(True)

        attack_data_iters[1].append(torch.clamp(b_x, min=0, max=1).cpu().detach().numpy())   # random noise

        cur_norm = init_norm

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0.01)

        for ii in tqdm(range(1, max_iter+1), desc='[SPLAT-{}]'.format(batch_idx)):
            delta.requires_grad_(True)

            predicted = torch.zeros_like(models[0](b_x+delta))
            for model in models:
                model.zero_grad()
                predicted = predicted + model(b_x)
            predicted = predicted / len(models)
            p = (torch.ones(predicted.shape) / predicted.shape[1]).to(predicted.device)
            cost = loss_fn(predicted, p)
            cost.backward(retain_graph=True)

            # renorming gradient
            delta.grad = delta.grad.renorm(p=2, dim=0, maxnorm=1)
            optimizer.step()

            # renorm the perturbation
            for idx in range(len(b_x)):
                delta[idx] = delta[[idx]].renorm(p=2, dim=0, maxnorm=cur_norm)[0]

            # divide the perturb norm bound by gamma
            if ii % per_iter == 0:
                adv = torch.clamp((b_x + delta), min=0, max=1)
                attack_data_iters[int(ii/per_iter) + 1].append(adv.cpu().detach().numpy())
                cur_norm = cur_norm * (1 - gamma)

            delta.detach_()
            scheduler.step()

    # to measure the time
    termi_time = time.time()
    print (' [SPLAT - L2] time taken for crafting 10k samples: {:.4f}'.format(termi_time - start_time))

    # return the data
    attack_data_iters = [np.vstack(attack_data) for attack_data in attack_data_iters]
    attack_labels = np.asarray(attack_labels)
    return attack_data_iters,  attack_labels