"""
    To compute GFLOPs (inference cost) and num params of a CNN or SDN
"""
# torch
import torch
import torch.nn as nn

# custom libs
import utils
import networks.SDNs.VGG_SDN as vgg_sdn
from networks.msdnet import ConvBasic_, ConvBN_, ConvDownNormal_, ConvNormal_, MSDNFirstLayer, MSDNLayer, \
    ParallelModule_, ClassifierModule_
from networks.msnet import BasicConv, ConvBN, ConvDownNormal, ConvNormal, Transition, MSNFirstLayer, MSNLayer, \
    ParallelModule, ClassifierModule


def count_conv2d(m, x, y):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops * m.groups

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])

def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])

def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])

def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_convbasic_(m, x, y):
    conv = m.net[0]
    cin = conv.in_channels
    cout = conv.out_channels
    kernel_ops = conv.kernel_size[0] * conv.kernel_size[1]
    output_elements = y.nelement()
    ops = cin * cout * kernel_ops * output_elements // conv.groups
    m.total_ops += torch.Tensor([ops])

def count_convbn_(m, x, y):
    ops = 0
    for layer in m.net:
        if isinstance(layer, nn.Conv2d):
            cin = layer.in_channels
            cout = layer.out_channels
            kernel_ops = layer.kernel_size[0] * layer.kernel_size[1]
            output_elements = y.nelement()
            ops += cin * cout * kernel_ops * output_elements // layer.groups
    m.total_ops += torch.Tensor([ops])

def count_convdownnormal_(m, x, y):
    ops = 0
    ops += m.conv_down.total_ops
    ops += m.conv_normal.total_ops
    m.total_ops += torch.Tensor([ops])

def count_convnormal_(m, x, y):
    ops = 0
    ops += m.conv_normal.total_ops
    m.total_ops += torch.Tensor([ops])

def count_msdnfirstlayer(m, x, y):
    ops = 0
    for layer in m.layers:
        ops += layer.total_ops
    m.total_ops += torch.Tensor([ops])

def count_msdnlayer(m, x, y):
    ops = 0
    for layer in m.layers:
        ops += layer.total_ops
    m.total_ops += torch.Tensor([ops])

def count_parallelmodule_(m, x, y):
    ops = 0
    for sub_module in m.m:
        ops += sub_module.total_ops
    m.total_ops += torch.Tensor([ops])

def count_classifiermodule_(m, x, y):
    ops = 0
    ops += m.m.total_ops
    if isinstance(m.linear, nn.Linear):
        ops += m.linear.weight.nelement() * y.size(0)
    m.total_ops += torch.Tensor([ops])


def count_basicconv(m, x, y):
    # BatchNorm2d operations: 2 * num_features
    bn_ops = 2 * x[0].size(1) * x[0].numel() / x[0].size(1)

    # ReLU operations: only activation
    relu_ops = x[0].numel()

    # Conv2d operations
    cin = m.conv.in_channels
    cout = m.conv.out_channels
    kh, kw = m.conv.kernel_size
    groups = m.conv.groups
    kernel_ops = kh * kw
    output_elements = y.nelement()
    conv_ops = cin * cout * kernel_ops * output_elements // groups

    total_ops = bn_ops + relu_ops + conv_ops
    m.total_ops += torch.Tensor([total_ops])

def count_convbn(m, x, y):
    ops = 0
    for layer in m.net:
        if isinstance(layer, nn.BatchNorm2d):
            bn_ops = 2 * layer.num_features * x[0].numel() / x[0].size(1)
            ops += bn_ops
        elif isinstance(layer, nn.ReLU):
            relu_ops = x[0].numel()
            ops += relu_ops
        elif isinstance(layer, nn.Conv2d):
            cin = layer.in_channels
            cout = layer.out_channels
            kh, kw = layer.kernel_size
            groups = layer.groups
            kernel_ops = kh * kw
            output_elements = y.nelement()
            conv_ops = cin * cout * kernel_ops * output_elements // groups
            ops += conv_ops
    m.total_ops += torch.Tensor([ops])

def count_convdownnormal(m, x, y):
    ops = 0
    ops += m.conv_down.total_ops
    ops += m.conv_up.total_ops
    ops += m.conv_normal.total_ops
    m.total_ops += torch.Tensor([ops])

def count_convnormal(m, x, y):
    ops = 0
    ops += m.conv_normal.total_ops
    m.total_ops += torch.Tensor([ops])

def count_transition(m, x, y):
    ops = 0
    ops += m.conv1_1x1.total_ops
    ops += m.conv2_3x3.total_ops
    if hasattr(m, 'conv_h'):
        ops += m.conv_h.total_ops
    ops += m.conv_s[0].total_ops
    ops += m.conv_s[1].total_ops

    # Global avg pool and fully connected layers
    out_channels = m.fc2.out_channels
    ops += out_channels * (m.fc1.in_channels / m.fc1.groups)
    ops += out_channels * m.fc2.in_channels
    m.total_ops += torch.Tensor([ops])

def count_msnfirstlayer(m, x, y):
    ops = 0
    for layer in m.layers:
        ops += layer.total_ops
    m.total_ops += torch.Tensor([ops])

def count_msnlayer(m, x, y):
    ops = 0
    for layer in m.layers:
        ops += layer.total_ops
    m.total_ops += torch.Tensor([ops])

def count_parallelmodule(m, x, y):
    ops = 0
    for sub_module in m.m:
        ops += sub_module.total_ops
    m.total_ops += torch.Tensor([ops])

def count_classifiermodule(m, x, y):
    ops = 0
    ops += m.m.total_ops
    ops += m.linear.weight.nelement() * y.size(0)
    m.total_ops += torch.Tensor([ops])


def profile_sdn(model, input_size, device, vgg=False):
    inp = (1, 3, input_size, input_size)
    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, ConvBasic_):
            m.register_forward_hook(count_convbasic_)
        elif isinstance(m, ConvBN_):
            m.register_forward_hook(count_convbn_)
        elif isinstance(m, ConvDownNormal_):
            m.register_forward_hook(count_convdownnormal_)
        elif isinstance(m, ConvNormal_):
            m.register_forward_hook(count_convnormal_)
        elif isinstance(m, MSDNFirstLayer):
            m.register_forward_hook(count_msdnfirstlayer)
        elif isinstance(m, MSDNLayer):
            m.register_forward_hook(count_msdnlayer)
        elif isinstance(m, ParallelModule_):
            m.register_forward_hook(count_parallelmodule_)
        elif isinstance(m, ClassifierModule_):
            m.register_forward_hook(count_classifiermodule_)
        elif isinstance(m, BasicConv):
            m.register_forward_hook(count_basicconv)
        elif isinstance(m, ConvBN):
            m.register_forward_hook(count_convbn)
        elif isinstance(m, ConvDownNormal):
            m.register_forward_hook(count_convdownnormal)
        elif isinstance(m, ConvNormal):
            m.register_forward_hook(count_convnormal)
        elif isinstance(m, Transition):
            m.register_forward_hook(count_transition)
        elif isinstance(m, MSNFirstLayer):
            m.register_forward_hook(count_msnfirstlayer)
        elif isinstance(m, MSNLayer):
            m.register_forward_hook(count_msnlayer)
        elif isinstance(m, ParallelModule):
            m.register_forward_hook(count_parallelmodule)
        elif isinstance(m, ClassifierModule):
            m.register_forward_hook(count_classifiermodule)
        elif isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass

    model.apply(add_hooks)

    x = torch.zeros(inp)
    x = x.to(device)
    model(x)

    output_total_ops = {}
    output_total_params = {}

    total_ops = 0
    total_params = 0

    if vgg:
        total_ops = 0
        total_params = 0

        cur_output_id = 0
        cur_output_layer_id = -10
        wait_for = -10

        vgg = False
        for layer_id, m in enumerate(model.modules()):
            if isinstance(m, utils.InternalClassifier):
                cur_output_layer_id = layer_id
            elif isinstance(m, vgg_sdn.FcBlockWOutput) and m.no_output == False:
                vgg = True
                cur_output_layer_id = layer_id

            if layer_id == cur_output_layer_id + 1:
                if vgg:
                    wait_for = 4
                elif isinstance(m, nn.Linear):
                    wait_for = 1
                else:
                    wait_for = 3

            if len(list(m.children())) > 0: continue

            total_ops += m.total_ops
            total_params += m.total_params

            if layer_id == cur_output_layer_id + wait_for:
                output_total_ops[cur_output_id] = total_ops.numpy()[0] / 1e9
                output_total_params[cur_output_id] = total_params.numpy()[0] / 1e6
                cur_output_id += 1

        output_total_ops[cur_output_id] = total_ops.numpy()[0] / 1e9
        output_total_params[cur_output_id] = total_params.numpy()[0] / 1e6
    else:
        classifier_index = 0

        for layer_id, m in enumerate(model.modules()):

            if len(list(m.children())) > 0: continue

            total_ops += m.total_ops
            total_params += m.total_params

            # Check if the module is a classifier module
            if 'Linear' in str(m):
                output_total_ops[classifier_index] = total_ops.numpy()[0] / 1e9
                output_total_params[classifier_index] = total_params.numpy()[0] / 1e6
                classifier_index += 1

    return output_total_ops, output_total_params



def profile(model, input_size, device):

    inp = (1, 3, input_size, input_size)
    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass
        else:
            #print("Not implemented for ", m)
            pass

    model.apply(add_hooks)

    x = torch.zeros(inp)
    x = x.to(device)
    model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_params += m.total_params
    total_ops = total_ops
    total_params = total_params

    return total_ops, total_params