# @title Autoload all modules
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# xia mian bie shan !!!!
import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
# import sampling2_parallel_pg


from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os

import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
# import controllable_generation
from utils import restore_checkpoint


sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils

import BD_sampl_cutcut_12ch as sampling_now
# sampling2_parallel
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from BD_sampl_cutcut_12ch import (ReverseDiffusionPredictor,
                             LangevinCorrector,
                             EulerMaruyamaPredictor,
                             AncestralSamplingPredictor,
                             NoneCorrector,
                             NonePredictor,
                             AnnealedLangevinDynamics)
import datasets
import os.path as osp

model_num = 'checkpoint.pth'
ckpt_filename = './exp/checkpoints_xzw_1/checkpoint_8.pth'

# @title Load the score-based model
sde = 'VESDE'  # @param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
    from configs.ve import SIAT_kdata_ncsnpp_test as configs  # 修改config

    # from configs.ve import bedroom_ncsnpp_continuous as configs  # 修改config
    # ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
    # ckpt_filename ='/home/lqg/桌面/ncsn++/score_sde_pytorch-SIAT_MRIRec_noise1_multichannel6/exp/checkpoints/checkpoint_33.pth'
    # ckpt_filename ='./exp/checkpoints/checkpoint_15.pth'

    config = configs.get_config()
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                N=config.model.num_scales)  ###################################  sde
    # sde = VESDE(sigma_min=0.01, sigma_max=10, N=100) ###################################  sde

    sampling_eps = 1e-5

batch_size = 1  # @param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0  # @param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())

# @title PC sampling
img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
predictor = ReverseDiffusionPredictor  # @param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector  # @param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.075  # 0.16 @param {"type": "number"}
n_steps = 2  # @param {"type": "integer"}
probability_flow = False  # @param {"type": "boolean"}
sampling_fn = sampling_now.get_pc_sampler(sde, shape, predictor, corrector,
                                          inverse_scaler, snr, n_steps=n_steps,
                                          probability_flow=probability_flow,
                                          continuous=config.training.continuous,
                                          eps=sampling_eps, device=config.device)
# sampling_fn = sampling2_parallel_pg.get_pc_sampler(sde, shape, predictor, corrector,
#                                       inverse_scaler, snr, n_steps=n_steps,
#                                       probability_flow=probability_flow,
#                                       continuous=config.training.continuous,
#                                       eps=sampling_eps, device=config.device)

x, n = sampling_fn(score_model)

'''
x = x.detach().cpu().numpy() # (1,3,256,256)

for ii in range(batch_size):      
  kw_real = (x[ii,0,:,:]+x[ii,2,:,:])/2
  kw_imag = x[ii,1,:,:]

  k_w = kw_real+1j*kw_imag
  image = np.fft.ifft2(k_w)
  max_ = np.max(np.abs(k_w))
  min_ = np.min(np.abs(k_w))
  save_kImage(k_w,'./result/sample/',"sample_"+"max="+str(max_)+"min="+str(min_)+".png")
  save_kImage(image,'./result/sample/',"sample_ifft2"+str(ii)+".png") 
'''
