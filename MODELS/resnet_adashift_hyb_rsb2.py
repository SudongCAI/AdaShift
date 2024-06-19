"""
from __future__ import absolute_import
import math

import torch
from torch.autograd import Variable 
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from functools import partial
from itertools import chain
from torch.utils.checkpoint import checkpoint

import torch.nn as nn
from torch.hub import load_state_dict_from_url

from einops import rearrange
from einops.layers.torch import Rearrange

import torch.linalg as linalg

from torch.autograd import Function
from torch.nn.init import calculate_gain

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.layers import DropBlock2d, AvgPool2dSame, BlurPool2d, GroupNorm, create_attn, get_attn, create_classifier

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg
"""


from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
#from timm.models.layers import trunc_normal_, DropPath
#from timm.models.layers import DropBlock2d, AvgPool2dSame, BlurPool2d, GroupNorm, create_attn, get_attn, create_classifier
#from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
#from timm.models.layers.helpers import to_2tuple
from timm.layers.helpers import to_2tuple

import math
import numpy as np
from numpy.random import shuffle as index_shuffle

from torch.autograd import Variable 
#from torch.nn.parameter import Parameter
from torch.hub import load_state_dict_from_url

from einops import rearrange
from einops.layers.torch import Rearrange

import torch.linalg as linalg

from torch.autograd import Function
from torch.nn.init import calculate_gain

from itertools import chain
from torch.utils.checkpoint import checkpoint


from typing import Any, Dict, List, Optional, Tuple, Type, Union
from timm.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, LayerType, create_attn, \
    get_attn, get_act_layer, get_norm_layer, create_classifier
from timm.layers import trunc_normal_, DropPath
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import register_model, generate_default_cfgs, register_model_deprecations




__all__ = ['ResNet', 'BasicBlock', 'Bottleneck']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }

default_cfg = {
    'silumod_mix_resnet14': _cfg(
        url='',
        interpolation='bicubic'),
    'silumod_mix_resnet26': _cfg(
        url='',
        interpolation='bicubic'),
    'silumod_mix_resnet50': _cfg(
        url='',
        interpolation='bicubic'),
    'silumod_mix_resnet101': _cfg(
        url='',
        interpolation='bicubic'),
    'silumod_mix_resnet152': _cfg(
        url='',
        interpolation='bicubic')
}



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)




def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)




########## with a learnable beta
class silumod(nn.Module):
    def __init__(self, planes):
        super(silumod, self).__init__()        
        
        self.stats = nn.AdaptiveAvgPool2d(1)

        # ch shift params
        self.proj = nn.Linear(planes, planes)
        self.norm = nn.LayerNorm(planes, eps=1e-06)
        nn.init.constant_(self.norm.weight, 0.01)
        nn.init.constant_(self.norm.bias, 0.)
        

        self.sign = 'silu mod'
    
    @torch.jit.script
    def compute(x, shift):
        return torch.sigmoid(shift + x) * x
    
    def forward(self, x):
        
        # adaptive translation        
        shift = self.norm(self.proj(self.stats(x).squeeze())).unsqueeze(-1).unsqueeze(-1) # b c 1 1
        
        return self.compute(x, shift)
    


###################
# *** Topsis activation
class silutopsis(nn.Module):
    def __init__(self, dim, rate_reduct=16, spatial=8, g=8, expand=1, \
                 use_scale=False, sp_ext=1):
        super(silutopsis, self).__init__()
        
        ## * TOPSIS module 
        # partition
        part = 7
        self.part = part # patch groups
        T = spatial // part
        self.T = T
        
        self.N = part * part
        
        ## query stats, key represents, value represents
        self.partition = nn.AvgPool2d(kernel_size=T, stride=T) if T>1 else nn.Identity()
        self.normq = nn.LayerNorm(dim, eps=1e-06)  
        self.normk = nn.LayerNorm(dim, eps=1e-06) 
        self.normv = nn.LayerNorm(dim, eps=1e-06)  
        
        self.softmax = nn.Softmax(dim=-1)
        
        # signature
        self.sign = 'easy ver, proj, w=1, dim-group=64' #(light comb w/ norm)
        
        # multi-head
        g = np.max([dim // 64, 1]) # min{g} = 1
        self.g = g
        
        
        # scale
        d = dim // self.g
        self.d = d
        self.scl = d**-0.5
        

        # LN rect
        if expand > 1:
            self.rect = nn.LayerNorm(dim, eps=1e-06)
            nn.init.constant_(self.rect.weight, 0.01)
            nn.init.constant_(self.rect.bias, 0.)
        else:
            self.rect = nn.Sequential(nn.Linear(dim, dim, bias=False),
                                      nn.LayerNorm(dim, eps=1e-06)
                                      )
            nn.init.constant_(self.rect[-1].weight, 1.)
            nn.init.constant_(self.rect[-1].bias, 0.)
    
    
        # init
        self.reset_parameters()
        

    @torch.jit.script
    def compute(x, shift): 
        return x * torch.sigmoid( shift + x )
    
    
    def forward(self, x): 
        
        b,c,h,w = x.shape
        
        N, g, part, T = self.N, self.g, self.part, self.T
        
        ## topsis calculate
        box = self.partition(x).flatten(2).transpose(1, 2) 
        
        ## ideal candidate
        # query (represents)
        q = self.normq(box).view(b, N, g, self.d).transpose(1, 2)  # b, g, N, d
        k = self.normk(box).view(b, N, g, self.d).transpose(1, 2)  # b, g, N, d
        v = self.normv(box).view(b, N, g, self.d).transpose(1, 2)  # b, g, N, d
        
        # similarity measure
        sim = self.scl * q @ k.transpose(-2, -1) # b g N d, b g N d -> b g N N 
        sim = self.softmax(sim)
        shift = sim @ v # b g i j, b g j d -> b g i d (i=N, j=N)
        shift = self.rect(shift.transpose(1, 2).reshape(b, N, c)).transpose(-2, -1).reshape(b, c, part, 1, part, 1)
        
        x = x.view(b, c, part, T, part, T)
        # local stats deco        
        return self.compute(x,  shift).view(b, c, h, w)


    def reset_parameters(self):        
        # conv and bn init
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):    
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

#################





########## with a learnable beta
### This one looks fine, but it's still worth trying other forms later
class silumodcomb(nn.Module):
    def __init__(self, dim_in, dim_in_r, dim_out, kernel_size=(1,1)):
        super(silumodcomb, self).__init__()        
        
        self.din = dim_in
        self.dinr = dim_in_r
        self.dout = dim_out
        self.ratio = dim_out/dim_in
        self.ratior = dim_out/dim_in_r
        
        self.stats = nn.AdaptiveAvgPool2d(1)
        
        # ch shift params        
        self.bias = nn.Parameter(0. * torch.ones(dim_out * 2), requires_grad=True) # 1-way bias
        self.norm = nn.LayerNorm(dim_out * 2, eps=1e-06)
        nn.init.constant_(self.norm.weight, 1e-06)
        nn.init.constant_(self.norm.bias, 0.) # init by 1. for cifar
        
        self.sign = 'silu mod comb (act3, shared projs ver, init w=1e-06)'
    
    
    @torch.jit.script
    def add(x1, x2):
        return x1 + x2
    
    @torch.jit.script
    def compute(x, st1, st2):
        return torch.sigmoid(st1 + st2 + x) * x
    
    
    def forward(self, x1, x2, proj1, proj2):
        
        b,c,h,w = x1.shape
        
        x = x1 + x2
        
        st1 = self.stats(x1).view(b, self.din, int(self.ratio)).mean(dim=-1, keepdim=True).unsqueeze(-1) # b d 1 1
        st1 = proj1(st1).squeeze() # b c 
        
        st2 = self.stats(x2).view(b, self.dinr, int(self.ratior)).mean(dim=-1, keepdim=True).unsqueeze(-1) if self.ratior > 1 else self.stats(x2) 
        st2 = proj2(st2).squeeze() # b c 
        
        st = torch.cat([st1, st2], dim=1) # b 2c     
        
        st1, st2 = self.norm(self.add(st, self.bias)).chunk(2, dim=1)
        st1, st2 = st1.unsqueeze(-1).unsqueeze(-1), st2.unsqueeze(-1).unsqueeze(-1)
        
        return self.compute(x, st1, st2)


####################







class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=silumod, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(first_planes)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(outplanes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return 



########### x version
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
            reduce_first=1, dilation=1, first_dilation=None, act_layer=silumod, norm_layer=nn.BatchNorm2d,
            attn_layer=None, aa_layer=None, drop_block=None, drop_path=None,
            rate_reduct=16, spatial=8, g=1):  # new args
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        #self.act1 = silumod(first_planes)
        self.act1 = silutopsis(first_planes, spatial=spatial*stride, expand = 1)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = silumod(width)
        #self.act2 = silutopsis(width, spatial=spatial, expand = 1)
        
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        #self.se = create_attn(attn_layer, outplanes)

        #self.act3 = silumod(outplanes)
        self.act3 = silumodcomb(width, inplanes, outplanes)
        self.identity = nn.Identity() #if downsample is None else None
        
        
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        #if self.se is not None:
        #    x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        
        #x += shortcut
        #x = self.act3(x)

        x = self.act3(x, shortcut, self.conv3, self.downsample[0]) if self.downsample is not None \
            else self.act3(x, shortcut, self.conv3, self.identity)

        return x





def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob=0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
        block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
        down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., 
        rate_reduct=[8,8,8,8], spatial=[56, 28, 14, 7], g=[1,1,1,1], sp_ext=[False, True, True, True], **kwargs): #new args
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db, rate_reduct, spatial, g, sp_ext) \
        in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate), \
                         rate_reduct, spatial, g, sp_ext)):
                                                                                              
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None, \
                    rate_reduct=rate_reduct, spatial=spatial, g=g, **block_kwargs))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.
    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.
    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.
    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.
    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.
    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`
    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return 


class ResNet(nn.Module):
    """ResNet
    Parameters
    avg_down : bool, default False, use average pooling for projection skip connection between stages/downsample.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0. Dropout probability before classifier, for training
    """

    def __init__(
            self, block, layers, num_classes=1000, in_chans=3, output_stride=32, global_pool='avg',
            cardinality=1, base_width=64, stem_width=64, stem_type='', replace_stem_pool=False, block_reduce_first=1,
            down_kernel_size=1, avg_down=False, act_layer=None, norm_layer=nn.BatchNorm2d, aa_layer=None,
            drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., zero_init_last=True, block_args=None,
            rate_reduct=[8,8,8,8], 
            pretrained_cfg=None,
            bn_tf=None): # new args
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = silutopsis(inplanes, spatial=112, expand=1)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True)
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        # Feature Blocks
        channels = [64, 128, 256, 512]
        rate_reduct = rate_reduct
        spatial = [56, 28, 14, 7]
        g = [1,1,1,1]
        sp_ext = [False, True, True, True]
        
        stage_modules, stage_feature_info = make_blocks(
            block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
            output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
            down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
            drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, 
            rate_reduct=rate_reduct, spatial=spatial, g=g, sp_ext=sp_ext,  **block_args) # new args
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        # init
        self.init_weights(zero_init_last=True) # close for sync-bn

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()



    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x) #* default setting
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x




"""
@register_model
def silumod_mix_resnet50(pretrained=False, in_22k=False, num_classes=1000, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    model.default_cfg = default_cfg['silumod_mix_resnet50']
    #if pretrained:
    #    state_dict = torch.hub.load_state_dict_from_url(
    #        url= model.default_cfg['url'], map_location="cpu", check_hash=True)
    #    model.load_state_dict(state_dict)
    return model
"""


def _create_resnet(variant, pretrained: bool = False, **kwargs) -> ResNet:
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


@register_model
def silumod_mix_resnet50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3],  **kwargs)
    model = _create_resnet('resnet50', pretrained, **dict(model_args, **kwargs))
    model.default_cfg = default_cfg['silumod_mix_resnet50']
    return model