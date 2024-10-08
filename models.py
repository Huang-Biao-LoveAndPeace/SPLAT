"""
    Contains the functions to define CNNs and SDNs,
    and also includes the hyper-parameters for training.
"""
# basic
import os, pickle
import torch

from networks import msdnet
# custom libs
from networks.CNNs.VGG import VGG
from networks.CNNs.ResNet import ResNet
from networks.CNNs.MobileNet import MobileNet

from networks.SDNs.VGG_SDN import VGG_SDN
from networks.SDNs.ResNet_SDN import ResNet_SDN
from networks.SDNs.MobileNet_SDN import MobileNet_SDN
from networks.msdnet import MSDNet
from networks.msnet import MSNet

from networks.inception import inception_v3
from networks.resnet import resnet18
from networks.vgg import vgg13_bn
from networks.densenet import densenet121
"""
    Store the network...
"""


def save_networks(network, parameters, storedir, savetype):
    # model_name example: cifar10_vgg16bn
    cnn_name = network+'_cnn'  # example: cifar10_vgg16bn_cnn
    sdn_name = network+'_sdn'  # example: cifar10_vgg16bn_sdn

    if 'c' in savetype:
        parameters['architecture'] = 'cnn'
        parameters['base_model']   = cnn_name
        nettype = parameters['network_type']

        if 'wideresnet' in nettype:
            model = WideResNet(parameters)
        elif 'resnet' in nettype:
            model = ResNet(parameters)
        elif 'vgg' in nettype:
            model = VGG(parameters)
        elif 'mobilenet' in nettype:
            model = MobileNet(parameters)

        # model: a child class of torch.nn.Model
        # model_params: dict containing hyperparameters
        # models_path: the path on disk to store models to
        # cnn_name: cifar10_vgg16bn_cnn for example
        save_model(model, cnn_name, parameters, storedir, epoch=0)

    if 's' in savetype:
        parameters['architecture'] = 'sdn'
        parameters['base_model']   = sdn_name
        nettype = parameters['network_type']

        if 'wideresnet' in nettype:
            model = WideResNet_SDN(parameters)
        elif 'resnet' in nettype:
            model = ResNet_SDN(parameters)
        elif 'vgg' in nettype:
            model = VGG_SDN(parameters)
        elif 'mobilenet' in nettype:
            model = MobileNet_SDN(parameters)

        # model: a child class of torch.nn.Model
        # model_params: dict containing hyperparameters
        # models_path: the path on disk to store models to
        # sdn_name: cifar10_vgg16bn_sdn for example
        save_model(model, sdn_name, parameters, storedir, epoch=0)

    return cnn_name, sdn_name


"""
    Create (init.) networks (for vanilla training)
"""
def create_vgg16bn(task, savedir, savetype, initialize=True):
    print (' [networks] Create VGG16-BN untrained {} model'.format(task))
    model_name   = '{}_vgg16bn'.format(task)
    model_params = load_task_params(task)

    # configure the architecture
    model_params['network_type'] = 'vgg16'
    if model_params['input_size'] == 32:
        model_params['fc_layers'] = [512, 512]
    elif model_params['input_size'] == 64:
        model_params['fc_layers'] = [2048, 1024]
    model_params['conv_channels']  = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    model_params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    model_params['conv_batch_norm'] = True
    model_params['init_weights'] = True
    model_params['augment_training'] = True

    # suppress the normalization - data will be in [0, 1]
    model_params['doNormalization'] = False

    # configure the augmentation of ICs
    model_params['add_output'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    load_lr_params(model_params)

    # only for initializing them, don't save
    if not initialize: return model_params

    # otherwise, save
    return save_networks(model_name, model_params, savedir, savetype)

def create_resnet56(task, savedir, savetype, initialize=True):
    print (' [networks] Create ResNet56 untrained {} model'.format(task))
    model_name   = '{}_resnet56'.format(task)
    model_params = load_task_params(task)

    # configure the architecture
    model_params['network_type'] = 'resnet56'
    model_params['block_type'] = 'basic'
    model_params['num_blocks'] = [9,9,9]
    model_params['augment_training'] = True
    model_params['init_weights'] = True

    # suppress the normalization - data will be in [0, 1]
    model_params['doNormalization'] = False

    # configure the augmentation of ICs
    model_params['add_output'] = [ \
        [1, 1, 1, 1, 1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 1, 1, 1, 1, 1]]
    load_lr_params(model_params)

    # only for initializing them, don't save
    if not initialize: return model_params

    # otherwise, save
    return save_networks(model_name, model_params, savedir, savetype)

def create_mobilenet(task, savedir, savetype, initialize=True):
    print (' [networks] Create MobileNet untrained {} model'.format(task))
    model_name   = '{}_mobilenet'.format(task)
    model_params = load_task_params(task)

    # configure the architecture
    model_params['network_type'] = 'mobilenet'
    model_params['cfg'] = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    model_params['augment_training'] = True
    model_params['init_weights'] = True

    # configure the augmentation of ICs
    model_params['add_output'] = [ \
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    load_lr_params(model_params)

    # only for initializing them, don't save
    if not initialize: return model_params

    # otherwise, save
    return save_networks(model_name, model_params, savedir, savetype)


"""
    Create (init.) networks (for AT)
"""
def create_vgg16bn_adv( \
    task, savedir, advcnn, attack, max_iter, epsilon, eps_step, save_type, initialize=True):
    print(' [networks] Creating ADN VGG16-BN untrained {} models...'.format(task))
    model_name = '{}_vgg16bn_{}_{}_{}_{}_{}'.format( \
        task, 'cnn-adv' if advcnn else 'cnn', attack, max_iter, epsilon, eps_step)
    model_params = load_task_params(task)

    # configure the architecture
    model_params['network_type'] = 'vgg16'
    model_params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    if model_params['input_size'] == 32:
        model_params['fc_layers'] = [512, 512]
    elif model_params['input_size'] == 64:
        model_params['fc_layers'] = [2048, 1024]
    model_params['conv_channels']  = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    model_params['conv_batch_norm'] = True
    model_params['init_weights'] = True
    model_params['augment_training'] = True

    # suppress the normalization - data will be in [0, 1]
    model_params['doNormalization'] = False

    # configure the augmentation of ICs
    model_params['add_output'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    load_lr_params(model_params)

    # add the adv. training parameters
    model_params['attack'] = attack
    model_params['iterations'] = max_iter
    model_params['eps_step'] = eps_step
    model_params['eps_max'] = epsilon

    # only for initializing them, don't save
    if not initialize: return model_params

    # otherwise, save
    return save_networks(model_name, model_params, savedir, save_type)

def create_resnet56_adv( \
    task, savedir, advcnn, attack, max_iter, epsilon, eps_step, save_type, initialize=True):
    print(' [networks] Creating ADN ResNet-56 untrained {} models...'.format(task))
    model_name = '{}_resnet56_{}_{}_{}_{}'.format( \
        task, task, 'cnn-adv' if advcnn else 'cnn', attack, max_iter, epsilon, eps_step)
    model_params = load_task_params(task)

    # configure the architecture
    model_params['network_type'] = 'resnet56'
    model_params['block_type'] = 'basic'
    model_params['num_blocks'] = [9,9,9]
    model_params['augment_training'] = True
    model_params['init_weights'] = True

    # suppress the normalization - data will be in [0, 1]
    model_params['doNormalization'] = False

    # configure the augmentation of ICs
    model_params['add_output'] = [ \
        [1, 1, 1, 1, 1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 1, 1, 1, 1, 1]]
    load_lr_params(model_params)

    # add the adv. training parameters
    model_params['attack'] = attack
    model_params['iterations'] = max_iter
    model_params['eps_step'] = eps_step
    model_params['eps_max'] = epsilon

    # only for initializing them, don't save
    if not initialize: return model_params

    # otherwise, save
    return save_networks(model_name, model_params, savedir, save_type)

def create_mobilenet_adv( \
    task, savedir, advcnn, attack, max_iter, epsilon, eps_step, save_type, initialize=True):
    print(' [networks] Creating ADN MobileNet untrained {} models...'.format(task))
    model_name = '{}_mobilenet_{}_{}_{}_{}'.format( \
        task, task, 'cnn-adv' if advcnn else 'cnn', attack, max_iter, epsilon, eps_step)
    model_params = load_task_params(task)

    # configure the architecture
    model_params['network_type'] = 'mobilenet'
    model_params['cfg'] = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    model_params['augment_training'] = True
    model_params['init_weights'] = True

    # suppress the normalization - data will be in [0, 1]
    model_params['doNormalization'] = False

    # configure the augmentation of ICs
    model_params['add_output'] = [ \
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    load_lr_params(model_params)

    # add the adv. training parameters
    model_params['attack'] = attack
    model_params['iterations'] = max_iter
    model_params['eps_step'] = eps_step
    model_params['eps_max'] = epsilon

    # only for initializing them, don't save
    if not initialize: return model_params

    # otherwise, save
    return save_networks(model_name, model_params, savedir, save_type)


"""
    Create (init.) networks (univeral functions)
"""
def create_vgg16bn_univ( \
    task, savedir, advcnn, advsdn, \
    attack, max_iter, epsilon, eps_step, save_type, initialize=True):
    print(' [networks] Creating (univ) VGG16-BN untrained {} models...'.format(task))
    model_name = '{}_vgg16bn_{}_{}_{}_{}_{}_{}'.format( \
        task, \
        'adv' if advcnn else 'none', \
        'adv' if advsdn else 'none', \
        attack, max_iter, epsilon, eps_step)
    model_params = load_task_params(task)

    # configure the architecture
    model_params['network_type'] = 'vgg16'
    model_params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    if model_params['input_size'] == 32:
        model_params['fc_layers'] = [512, 512]
    elif model_params['input_size'] == 64:
        model_params['fc_layers'] = [2048, 1024]
    model_params['conv_channels']  = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    model_params['conv_batch_norm'] = True
    model_params['init_weights'] = True
    model_params['augment_training'] = True

    # suppress the normalization - data will be in [0, 1]
    model_params['doNormalization'] = False

    # configure the augmentation of ICs
    model_params['add_output'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    load_lr_params(model_params)

    # add the adv. training parameters
    model_params['attack'] = attack
    model_params['iterations'] = max_iter
    model_params['eps_step'] = eps_step
    model_params['eps_max'] = epsilon

    # only for initializing them, don't save
    if not initialize: return model_params

    # otherwise, save
    return save_networks(model_name, model_params, savedir, save_type)

def create_resnet56_univ( \
    task, savedir, advcnn, advsdn, \
    attack, max_iter, epsilon, eps_step, save_type, initialize=True):
    print(' [networks] Creating (univ) ResNet-56 untrained {} models...'.format(task))
    model_name = '{}_vgg16bn_{}_{}_{}_{}_{}_{}'.format( \
        task, \
        'adv' if advcnn else 'none', \
        'adv' if advsdn else 'none', \
        attack, max_iter, epsilon, eps_step)
    model_params = load_task_params(task)

    # configure the architecture
    model_params['network_type'] = 'resnet56'
    model_params['block_type'] = 'basic'
    model_params['num_blocks'] = [9,9,9]
    model_params['augment_training'] = True
    model_params['init_weights'] = True

    # suppress the normalization - data will be in [0, 1]
    model_params['doNormalization'] = False

    # configure the augmentation of ICs
    model_params['add_output'] = [ \
        [1, 1, 1, 1, 1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 1, 1, 1, 1, 1]]
    load_lr_params(model_params)

    # add the adv. training parameters
    model_params['attack'] = attack
    model_params['iterations'] = max_iter
    model_params['eps_step'] = eps_step
    model_params['eps_max'] = epsilon

    # only for initializing them, don't save
    if not initialize: return model_params

    # otherwise, save
    return save_networks(model_name, model_params, savedir, save_type)

def create_mobilenet_univ( \
    task, savedir, advcnn, advsdn, \
    attack, max_iter, epsilon, eps_step, save_type, initialize=True):
    print(' [networks] Creating (univ) MobileNet untrained {} models...'.format(task))
    model_name = '{}_vgg16bn_{}_{}_{}_{}_{}_{}'.format( \
        task, \
        'adv' if advcnn else 'none', \
        'adv' if advsdn else 'none', \
        attack, max_iter, epsilon, eps_step)
    model_params = load_task_params(task)

    # configure the architecture
    model_params['network_type'] = 'mobilenet'
    model_params['cfg'] = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    model_params['augment_training'] = True
    model_params['init_weights'] = True

    # suppress the normalization - data will be in [0, 1]
    model_params['doNormalization'] = False

    # configure the augmentation of ICs
    model_params['add_output'] = [ \
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    load_lr_params(model_params)

    # add the adv. training parameters
    model_params['attack'] = attack
    model_params['iterations'] = max_iter
    model_params['eps_step'] = eps_step
    model_params['eps_max'] = epsilon

    # only for initializing them, don't save
    if not initialize: return model_params

    # otherwise, save
    return save_networks(model_name, model_params, savedir, save_type)


"""
    Parameter set-ups
"""
def load_task_params(task):
    if task == 'cifar10':
        return cifar10_params()
    elif task == 'cifar100':
        return cifar100_params()
    elif task == 'tinyimagenet':
        return tiny_imagenet_params()
    elif task == 'tinyimagenetsub':
        return tiny_imagenet_subset_params()

def cifar10_params():
    model_params = {}
    model_params['task'] = 'cifar10'
    model_params['input_size'] = 32
    model_params['num_classes'] = 10
    return model_params

def cifar100_params():
    model_params = {}
    model_params['task'] = 'cifar100'
    model_params['input_size'] = 32
    model_params['num_classes'] = 100
    return model_params

def tiny_imagenet_params():
    model_params = {}
    model_params['task'] = 'tinyimagenet'
    model_params['input_size'] = 64
    model_params['num_classes'] = 200
    return model_params

def tiny_imagenet_subset_params():
    model_params = {}
    model_params['task'] = 'tinyimagenetsub'
    model_params['input_size'] = 64
    model_params['num_classes'] = 10
    return model_params

"""DeepSLoth代码的结果
def load_lr_params(model_params):
    # vanilla models
    model_params['epochs'] = 100 # ionut: original 100
    model_params['learning_rate'] = 0.01
    model_params['gammas'] = [0.1, 0.1]
    model_params['momentum'] = 0.9
    model_params['milestones'] = [10, 20]

    # control the weight decay
    if 'vgg' in model_params['network_type'] \
        or 'wideresnet' in model_params['network_type']:
        model_params['weight_decay'] = 0.0005
    else:
        model_params['weight_decay'] = 0.0001

    # SDN models (ic-only cases)
    model_params['ic_only'] = {}
    model_params['ic_only']['learning_rate'] = 0.001
    model_params['ic_only']['epochs'] = 25 # ionut: original 25
    model_params['ic_only']['milestones'] = [15] # ionut: original [15]
    model_params['ic_only']['gammas'] = [0.1]
    # done.
"""

#SDN代码的参数设置
def load_lr_params(model_params):
    model_params['momentum'] = 0.9

    network_type = model_params['network_type']

    if 'vgg' in network_type or 'wideresnet' in network_type:
        model_params['weight_decay'] = 0.0005

    else:
        model_params['weight_decay'] = 0.0001

    model_params['learning_rate'] = 0.1
    model_params['epochs'] = 100
    model_params['milestones'] = [35, 60, 85]
    model_params['gammas'] = [0.1, 0.1, 0.1]

    # SDN ic_only training params
    model_params['ic_only'] = {}
    model_params['ic_only']['learning_rate'] = 0.001  # lr for full network training after sdn modification
    model_params['ic_only']['epochs'] = 25
    model_params['ic_only']['milestones'] = [15]
    model_params['ic_only']['gammas'] = [0.1]

"""
    Store function
"""
def save_model(network, netname, parameters, storedir, epoch=-1):
    if not os.path.exists(storedir): os.makedirs(storedir)
    netpath = os.path.join(storedir, netname)
    if not os.path.exists(netpath): os.makedirs(netpath)

    # epoch == 0 is the untrained network, epoch == -1 is the last
    if epoch == 0:
        path = os.path.join(netpath, 'untrained')
        params_path = os.path.join(netpath, 'parameters_untrained')
    elif epoch == -1:
        path = os.path.join(netpath, 'last')
        params_path = os.path.join(netpath, 'parameters_last')
    else:
        path = os.path.join(netpath, str(epoch))
        params_path = os.path.join(netpath, f'parameters_{epoch}')

    # store the pytorch model
    torch.save(network.state_dict(), path)

    # store the parameters
    if parameters is not None:
        with open(params_path, 'wb') as outfile:
            pickle.dump(parameters, outfile, pickle.HIGHEST_PROTOCOL)
    print(f'[SAVE] The model was saved to {netpath}')
    # done.

def load_params(netpath, netname, epoch=0):
    params_path = os.path.join(netpath, netname)
    if epoch == 0:
        params_path = os.path.join(params_path, 'parameters_untrained')
    else:
        params_path = os.path.join(params_path, 'parameters_last')

    # load
    with open(params_path, 'rb') as infile:
        model_params = pickle.load(infile)
    return model_params

def load_cnn_parameters(dataset, network):
    if network == 'vgg16':
        return create_vgg16bn(dataset, None, None, initialize=False)
    elif network == 'resnet56':
        return create_resnet56(dataset, None, None, initialize=False)
    elif network == 'mobilenet':
        return create_mobilenet(dataset, None, None, initialize=False)
    # done.

def get_args(model_name):
    # 模拟的配置文件，根据模型名称提供参数
    if 'cifar10' in model_name:
        data = 'cifar10'
    elif 'cifar100' in model_name:
        data = 'cifar100'
    elif 'tinyimagenet' in model_name:
        data = 'tinyimagenet'
    elif 'tinyimagenetsub' in model_name:
        data = 'tinyimagenet'
    else:
        print(" unknown dataset")
        return
    config = {
        'msdnet': {
            'arch': 'msdnet',
            'nBlocks': 7,
            'nChannels': 16,
            'base': 4,
            'stepmode': 'lingrow',
            'step': 2,
            'growthRate': 6,
            'grFactor': [1, 2, 4],
            'prune': 'max',
            'bnFactor':[1, 2, 4],
            'bottleneck': True,
            'data': data,
            'nScales': 3,
            'reduction': 0.5,
        },
        'msnet': {
            'data': data,
            'reduction': 0.5,
            'arch': 'msnet',
            'nBlocks': 7,
            'stepmode': 'lin_grow',
            'prune': 'max',
            'bottleneck': True,
            'compress_factor': 0.5,
            'step': [4, 4, 6, 2, 2, 10, 2],
            'nChannels': 12,
            'growthRate': 6,
            'grFactor': [1, 2, 4],
            'bnFactor': [1, 2, 4],
            'transFactor': [2, 5],
            'nScales': 3,
        }
    }

    # 提取模型名称中的关键字以确定模型类型
    if 'msdnet' in model_name:
        args = config['msdnet']
    elif 'msnet' in model_name:
        args = config['msnet']
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # 将字典转换为对象，以便通过属性访问参数
    class Args:
        pass

    args_obj = Args()
    for key, value in args.items():
        setattr(args_obj, key, value)

    return args_obj

def load_model(models_path, model_name, epoch=0, device='cpu'):
    num_classes = -1
    if 'cifar100' in model_name:
        num_classes = 100
    elif 'cifar10' in model_name:
        num_classes = 10
    elif 'tinyimagenetsub' in model_name:
        num_classes = 10
    elif 'tinyimagenet' in model_name:
        num_classes = 200

    model_params = load_params(models_path, model_name, epoch)

    architecture = 'empty' if 'architecture' not in model_params else model_params['architecture']
    network_type = 'None'
    if 'cnn' in model_name or '_sdn' in model_name:
        network_type = model_params['network_type']

    if architecture == 'cnn' or ('cnn' in model_name):
        if 'resnet' in network_type:
            model = ResNet(model_params)
        elif 'vgg' in network_type:
            model = VGG(model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet(model_params)

    else:
        if 'resnet56' in model_name:
            model = ResNet_SDN(model_params)
        elif 'vgg16bn' in model_name:
            model = VGG_SDN(model_params)
        elif 'mobilenet' in model_name:
            model = MobileNet_SDN(model_params)
        elif 'msdnet' in model_name:
            args = get_args(model_name)
            model = MSDNet(args)
        elif 'msnet' in model_name:
            args = get_args(model_name)
            model = MSNet(args)
        elif 'densenet121' in model_name:
            model = densenet121(num_classes = num_classes)
        elif 'inception_v3' in model_name:
            model = densenet121(num_classes = num_classes)
        elif 'vgg13_bn' in model_name:
            model = vgg13_bn(num_classes = num_classes)
        elif 'resnet18' in model_name:
            model = resnet18(num_classes = num_classes)

    network_path = os.path.join(models_path, model_name)
    if epoch == 0:
        # untrained model
        load_path = os.path.join(network_path, 'untrained')
    elif epoch == -1:
        # last model
        load_path = os.path.join(network_path, 'last')
    else:
        load_path = os.path.join(network_path, str(epoch))

    model = model.to(device)
    dic = torch.load(load_path, map_location=device)
    if 'msdnet' in model_name or 'msnet' in model_name:
        new_state_dict = {}
        for key, value in dic.items():
            # 假设前缀是 'module.'
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(dic, strict=True)

    return model, model_params

def load_cnn(sdn):
    if isinstance(sdn, VGG_SDN):
        return VGG
    elif isinstance(sdn, ResNet_SDN):
        return ResNet
    elif isinstance(sdn, MobileNet_SDN):
        return MobileNet

def load_sdn(cnn):
    if isinstance(cnn, VGG):
        return VGG_SDN
    elif isinstance(cnn, ResNet):
        return ResNet_SDN
    elif isinstance(cnn, MobileNet):
        return MobileNet_SDN
