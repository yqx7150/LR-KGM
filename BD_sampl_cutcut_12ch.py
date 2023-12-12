# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools
import os

import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
# from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import os.path as osp
import matplotlib.pyplot as plt
import scipy.io as io
import math


_CORRECTORS = {}
_PREDICTORS = {}

# typezl = 'testMe/'
# cut0 = 512
# cut1 = 512
# cut2 = 512

def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(sde=sde,
                                      shape=shape,
                                      inverse_scaler=inverse_scaler,
                                      denoise=config.sampling.noise_removal,
                                      eps=eps,
                                      device=config.device)
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     inverse_scaler=inverse_scaler,
                                     snr=config.sampling.snr,
                                     n_steps=config.sampling.n_steps_each,
                                     probability_flow=config.sampling.probability_flow,
                                     continuous=config.training.continuous,
                                     denoise=config.sampling.noise_removal,
                                     eps=eps,
                                     device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    # Alogrithm 2
    def update_fn(self, x, t):
        # ============================
        # file_path='/home/lqg/桌面/TZJ_SDE/input_data/SIAT_test_image31/test_data_0'+str(5)+'.mat'
        # ori_data = io.loadmat(file_path)['Img']
        # ori_data = ori_data/np.max(abs(ori_data))
        # mask = io.loadmat('/home/lqg/桌面/TZJ_SDE/input_data/mask/mask_radial_030.mat')['mask_radial_030']
        # mask = io.loadmat('/home/lqg/桌面/TZJ_SDE/input_data/mask/random_mask_r3_256.mat')['mask']
        # mask = io.loadmat('./input_data/mask/poisson/2.mat')['mask']
        # mask = io.loadmat('/home/lqg/桌面/TZJ_SDE/input_data/mask/random2D/2.mat')['mask']

        # mask = np.fft.fftshift(io.loadmat('/home/lqg/桌面/TZJ_SDE/input_data/mask/mask_cart85.mat')['mask_cart85'])
        # temp = np.random.uniform(0,1,size=(256,256))
        # mask = (temp>=0.9)+0
        # mask[127-25:127+25,127-25:127+25] = 1

        # weight = io.loadmat('/home/lqg/桌面/TZJ_SDE/input_data/weight1.mat')['weight']
        # Kdata=np.fft.fftshift(np.fft.fft2(ori_data))
        # ori_data = np.fft.ifft2(Kdata)
        # Ksample=np.multiply(mask,Kdata)
        # ============================
        f, G = self.rsde.discretize(x, t)  # 3
        z = torch.randn_like(x)  # 4
        x_mean = x - f  # 3
        x = x_mean + G[:, None, None, None] * z  # 5

        return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x1, x2, x3, x_mean, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        # Algorithm 4
        for i in range(n_steps):
            grad1 = score_fn(x1, t)  # 5
            grad2 = score_fn(x2, t)  # 5
            grad3 = score_fn(x3, t)  # 5

            noise1 = torch.randn_like(x1)  # 4
            noise2 = torch.randn_like(x2)  # 4
            noise3 = torch.randn_like(x3)  # 4

            grad_norm1 = torch.norm(grad1.reshape(grad1.shape[0], -1), dim=-1).mean()
            noise_norm1 = torch.norm(noise1.reshape(noise1.shape[0], -1), dim=-1).mean()
            grad_norm2 = torch.norm(grad2.reshape(grad2.shape[0], -1), dim=-1).mean()
            noise_norm2 = torch.norm(noise2.reshape(noise2.shape[0], -1), dim=-1).mean()
            grad_norm3 = torch.norm(grad3.reshape(grad3.shape[0], -1), dim=-1).mean()
            noise_norm3 = torch.norm(noise3.reshape(noise3.shape[0], -1), dim=-1).mean()

            grad_norm = (grad_norm1 + grad_norm2 + grad_norm3) / 3.0
            noise_norm = (noise_norm1 + noise_norm2 + noise_norm3) / 3.0

            step_size = (2 * alpha) * ((target_snr * noise_norm / grad_norm) ** 2)  # 6

            x_mean = x_mean + step_size[:, None, None, None] * (grad1 + grad2 + grad3) / 3.0  # 7
            # x_mean = x_mean + step_size[:, None, None, None] * grad1 # 7

            x1 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise1  # 7
            x2 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise2  # 7
            x3 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise3  # 7

        return x1, x2, x3, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]
        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x1, x2, x3, x_mean, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x1, x2, x3, x_mean, t)



def write_kdata(Kdata, name):
    temp = np.log(1 + abs(Kdata))
    plt.axis('off')
    plt.imshow(abs(temp), cmap='gray')
    plt.savefig(osp.join('./result/', name), transparent=True, dpi=128, pad_inches=0, bbox_inches='tight')


def write_Data(model_num, psnr, ssim,typezl):
    filedir = "result.txt"
    with open(osp.join('./result/'+ typezl, filedir), "w+") as f:  # a+
        f.writelines(str(model_num) + ' ' + '[' + str(round(psnr, 2)) + ' ' + str(round(ssim, 4)) + ']')
        f.write('\n')


def write_Data2(psnr, ssim,typezl):
    filedir = "PC.txt"
    with open(osp.join('./result/'+typezl, filedir), "a+") as f:  # a+
        f.writelines('[' + str(round(psnr, 2)) + ' ' + str(round(ssim, 4)) + ']')
        f.write('\n')


def write_images(x, image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)


def k2wgt(X, W):
    Y = np.multiply(X, W)
    return Y


def wgt2k(X, W, DC):
    Y = np.multiply(X, 1. / W)
    Y[W == 0] = DC[W == 0]
    return Y


def im2row(im, winSize):
    size = (im).shape
    out = np.zeros(((size[0] - winSize[0] + 1) * (size[1] - winSize[1] + 1), winSize[0] * winSize[1], size[2]),
                   dtype=np.complex64)
    count = -1
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1
            temp1 = im[x:(size[0] - winSize[0] + x + 1), y:(size[1] - winSize[1] + y + 1), :]
            temp2 = np.reshape(temp1, [(size[0] - winSize[0] + 1) * (size[1] - winSize[1] + 1), 1, size[2]], order='F')
            out[:, count, :] = np.squeeze(temp2)  # MATLAB reshape

    return out


def row2im(mtx, size_data, winSize):
    size_mtx = mtx.shape  # (63001, 36, 8)
    sx = size_data[0]  # 256
    sy = size_data[1]  # 256
    sz = size_mtx[2]  # 8

    res = np.zeros((sx, sy, sz), dtype=np.complex64)
    W = np.zeros((sx, sy, sz), dtype=np.complex64)
    out = np.zeros((sx, sy, sz), dtype=np.complex64)
    count = -1

    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1
            res[x: sx - winSize[0] + x + 1, y: sy - winSize[1] + y + 1, :] = res[x: sx - winSize[0] + x + 1,
                                                                             y: sy - winSize[1] + y + 1,
                                                                             :] + np.reshape(
                np.squeeze(mtx[:, count, :]), [sx - winSize[0] + 1, sy - winSize[1] + 1, sz], order='F')
            W[x: sx - winSize[0] + x + 1, y: sy - winSize[1] + y + 1, :] = W[x: sx - winSize[0] + x + 1,
                                                                           y: sy - winSize[1] + y + 1, :] + 1

    out = np.multiply(res, 1. / W)
    return out



import copy
import torch as tc
def hosvd(x):
    ndim = x.ndim
    U = list()
    lm = list()
    x = tc.from_numpy(x)
    # 计算各个指标的键约化矩阵（此处x为实矩阵）
    for n in range(ndim):
        index = list(range(ndim))
        index.pop(n)
        # 每次从n个指标中去除n-1个指标，随后进行张量积操作
        _mat = tc.tensordot(x, x, [index, index])
        # torch.symeig求对称矩阵的特征值和特征向量
        _lm, _U = tc.symeig(_mat, eigenvectors=True)
        lm.append(_lm.numpy())
        # _U[:,int(_U.shape[1]/10):_U.shape[1]]=0 useless
        #
        # if _U.shape[1]==121:
        #     _U = _U[:100,:]
        # if _U.shape[1] == 512:
        #     _U = _U[:480, :]

        U.append(_U)

    # 计算核张量G
    G = tucker_product(x, U)
    U1 = [u.numpy() for u in U]
    return G, U1, lm
def tucker_product(x, U, dim=1):
    ndim = x.ndim
    if type(x) is not tc.Tensor:
        x = tc.from_numpy(x)

    U1 = list()
    for n in range(len(U)):
        if type(U[n]) is not tc.Tensor:
            U1.append(tc.from_numpy(U[n]))
        else:
            U1.append(U[n])

    ind_x = ''
    for n in range(ndim):
        ind_x += chr(97 + n)
    ind_x1 = ''
    for n in range(ndim):
        ind_x1 += chr(97 + ndim + n)
    contract_eq = copy.deepcopy(ind_x)
    for n in range(ndim):
        if dim == 0:
            contract_eq += ',' + ind_x[n] + ind_x1[n]
        else:
            contract_eq += ',' + ind_x1[n] + ind_x[n]
    contract_eq += '->' + ind_x1
    G = tc.einsum(contract_eq, [x] + U1)
    G = G.numpy()
    return G
def hosvd_zl(tensorAll):

    tensorReal = np.real(tensorAll)
    tensorIma = np.imag(tensorAll)

    Core, V, LM = hosvd(tensorReal)
    tensorReal_svd = tucker_product(Core, V, dim=0)
    error1 = np.linalg.norm(tensorReal - tensorReal_svd)
    # print('Turker分解误差实数 = ' + str(error1))
    Core2, V2, LM2 = hosvd(tensorIma)
    tensorIma_svd = tucker_product(Core2, V2, dim=0)
    error2 = np.linalg.norm(tensorIma - tensorIma_svd)
    # print('Turker分解误差虚数 = ' + str(error2))
    tensorAll_svd = tensorReal_svd + tensorIma_svd * 1j
    error3 = np.linalg.norm(tensorAll - tensorAll_svd)
    print('Turker分解误差合并  = ' + str(error3))

    tensorAll_sos = np.sqrt(np.sum(np.abs((tensorAll) ** 2), axis=2))
    tensorAll_svd_sos = np.sqrt(np.sum(np.abs((tensorAll_svd) ** 2), axis=2))

    # plt.subplot(1, 3, 1)
    # plt.imshow(255 * abs(tensorAll_sos), cmap='gray')
    # plt.title("ori")
    # plt.subplot(1, 3, 2)
    # plt.imshow(255 * abs(tensorAll_svd_sos), cmap='gray')
    # plt.title("svd")

    plt.show()
    return tensorAll_svd
def svd_zl(input, cutSize):
    # SVD ===============================================
    svd_input = torch.tensor(input, dtype=torch.complex64)
    U, S, V = torch.svd(svd_input)
    S = torch.diag(S)

    U = np.array(U, dtype=np.complex64)  # 61952*512
    S = np.array(S, dtype=np.complex64)  # 512*512
    V = np.array(V, dtype=np.complex64)  # 512*512

    # zero jie duan
    uu = U[:, 0:math.floor(cutSize)]  # (63001, 64)
    ss = S[0:math.floor(cutSize),
       0:math.floor(cutSize)]  # (64, 64)
    vv = V[:, 0:math.floor(cutSize)]  # (288,64)

    A_svd = np.dot(np.dot(uu, ss), vv.T)
    # get A_svd -- 61952*512  cutSize=512 ======================================
    return A_svd
def svd_facker(input):
    return input


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
    """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def pc_sampler(model):
        """ The PC sampler funciton.
    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
        with torch.no_grad():

            cut_1=1411
            cut0 = 64 # int(64 * 0.6) # 38
            cut1 = 100
            cut2 = 768
            typezl = 'cutTest3/poisson_b/' +str(cut_1)+'_'+ str(cut0) + '_' + str(cut1) + '_'+ str(cut2) + '/'
            isExists = os.path.exists('./result/'+typezl)
            # 判断结果
            if not isExists:
                # 如果不存在则创建目录
                # 创建目录操作函数
                os.makedirs('result/'+typezl)
                print('result/'+typezl + ' 创建成功')
            else:
                print('result/'+typezl + ' 目录已存在')


            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
            # ori_data get
            coil = 12
            # file_path = '../parallel_inputdata/contract_data_8h/data1_GE_brain.mat'
            file_path = './input_data/test_GT_12ch/11.mat'
            # file_path = '/home/lqg/xzw/input_data/8ch/brain_8ch_ori.mat'
            ori_data = np.zeros([256, 256, coil], dtype=np.complex64)
            ori_data = io.loadmat(file_path)['img']
            ori_data = ori_data / np.max(abs(ori_data))
            ori_data = np.swapaxes(ori_data, 0, 2)
            ori_data = np.swapaxes(ori_data, 1, 2)
            ori_data_sos = np.sqrt(np.sum(np.square(np.abs(ori_data)), axis=0))
            write_images(abs(ori_data_sos), osp.join('./result/' + typezl + 'ori' + '.png'))
            # mask get
            mask = np.zeros((coil, 256, 256))
            # mask_item = io.loadmat('../parallel_inputdata/contract_mask/random2D/6.mat')['mask']
            mask_item = io.loadmat('./input_data/mask/all_mask/Poisson_R4_24X24.mat')['mask']
            # mask_item = io.loadmat('/home/lqg/xzw/input_data/mask/poisson/mask_possion_R8.mat')['mask']
            # mask_item = io.loadmat('/home/lqg/xzw/input_data/mask/Partial/partialFourier2D_R3.mat')['mask']
            for i in range(coil):
                mask[i, :, :] = mask_item

            # temp = np.random.uniform(0,1,size=(256,256))
            # mask = (temp >= 0.9) + 0
            # mask[127 - 25:127 + 25, 127 - 25:127 + 25] = 1
            # print(np.sum(mask_item) / 65536)
            # plt.imshow(abs(mask), cmap='gray')
            # plt.show()
            # assert 0




            write_images(abs(mask_item), osp.join('./result/' + typezl + 'mask' + '.png'))
            # weight get
            # ww = io.loadmat('input_data/weight1_GEBrain.mat')['weight']
            ww = io.loadmat('./input_data/GE8ch_W1/weight1_GEBrain4.mat')['weight']  # 40.43 0.9593
            weight = np.zeros((coil, 256, 256))
            for i in range(coil):
                weight[i, :, :] = ww
            # ori to kspace+mask+weight and zeroFill get
            Kdata = np.zeros((coil, 256, 256), dtype=np.complex64)
            Ksample = np.zeros((coil, 256, 256), dtype=np.complex64)
            zeorfilled_data = np.zeros((coil, 256, 256), dtype=np.complex64)
            k_w = np.zeros((coil, 256, 256), dtype=np.complex64)
            for i in range(coil):
                Kdata[i, :, :] = np.fft.fftshift(np.fft.fft2(ori_data[i, :, :]))
                Ksample[i, :, :] = np.multiply(mask[i, :, :], Kdata[i, :, :])
                k_w[i, :, :] = k2wgt(Ksample[i, :, :], weight[i, :, :])
                zeorfilled_data[i, :, :] = np.fft.ifft2(Ksample[i, :, :])
            zeorfilled_data_sos = np.sqrt(np.sum(np.square(np.abs(zeorfilled_data)), axis=0))
            # zero fill psnr
            ori_data_sos = ori_data_sos / np.max(np.abs(ori_data_sos))
            zeorfilled_data_sos = zeorfilled_data_sos / np.max(np.abs(zeorfilled_data_sos))
            print('k_w max:', np.max(np.abs(k_w)))
            psnr_zero = compare_psnr(255 * abs(zeorfilled_data_sos), 255 * abs(ori_data_sos), data_range=255)
            ssim_zero = compare_ssim(abs(zeorfilled_data_sos), abs(ori_data_sos), data_range=1)
            print('psnr_zero: ', psnr_zero, 'ssim_zero: ', ssim_zero)
            write_images(abs(zeorfilled_data_sos), osp.join('./result/' + typezl + 'Zeorfilled_' + str(round(psnr_zero, 2)) + str(round(ssim_zero, 4)) + '.png'))
            io.savemat(osp.join('./result/' +typezl + 'zeorfilled.mat'), {'zeorfilled': zeorfilled_data})
            # -- now we get k_w:coil*256*256
            k_w = k_w.transpose(1, 2, 0)

            # -- now we get k_w:256*256*coil

            # set hankel params
            ksize = [8, 8]
            # hankel
            hankel = im2row(k_w, ksize)
            size_temp = hankel.shape
            # print(size_temp)
            # assert 0
            # -- hankel size_temp:(62001, 64, 12) / 47616768
            A = np.reshape(hankel, [size_temp[0], size_temp[1] * size_temp[2]], order='F')
            # -- A (62001, 768)
            # cut 62001*512 to 80*768*768
            A_temp = np.zeros((80, 768, 768))
            ans_1 = np.array(A_temp, dtype=np.complex64)
            for i in range(80):  # diu 561
              cut = A[768 * i:768 * (i + 1),:]
              ans_1[i, :, :] = cut
            # get ans_1 80*768*768 = 47185920  / ps: 62001*768 = 47616768
            ans = np.concatenate((np.real(ans_1), np.imag(ans_1)), 0)
            # print(ans.shape)
            # assert 0
            x_mean = torch.tensor(ans, dtype=torch.float32).cuda().unsqueeze(0)
            # get x_mean 160,768,768 -- from ans_1 ...


            x1 = x_mean
            x2 = x_mean
            x3 = x_mean

            max_psnr = 0
            max_ssim = 0
            # set svd param
            wnthresh = 0.6 # 1.2 0.6
            size_data = [256, 256, 12]

            # start

            for i in range(sde.N):
                print('======== ', i)
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t

                ##======================================================= Predictor
                # x_mean 160*768*768 tensor float
                x, x_mean = predictor_update_fn(x_mean, vec_t, model=model)
                x_mean = x_mean.cpu().numpy().squeeze(0)  # (8,6,256,256)
                A_complex = x_mean[0:80, :, :] + 1j * x_mean[80:, :, :]


                def unfoldTo2D(thing_3D, index):
                    # A's size -- 80*768*768
                    size_3D = thing_3D.shape
                    thing_2D = np.zeros((size_3D[0] * size_3D[1], size_3D[2]), dtype=np.complex64)
                    if index == 0:
                        for i in range(size_3D[0]):
                            thing_2D[size_3D[1] * i:size_3D[1] * (i + 1), :] = thing_3D[i, :, :]
                    elif index == 1:
                        for i in range(size_3D[1]):
                            thing_2D[size_3D[0] * i:size_3D[0] * (i + 1), :] = thing_3D[:, i, :]
                    elif index == 2:
                        for i in range(size_3D[2]):
                            thing_2D[size_3D[0] * i:size_3D[0] * (i + 1), :] = thing_3D[:, :, i]
                    else:
                        return -996

                    return thing_2D

                def pileTo3D(thing_2D, index, size_3D=(80, 768, 768)):
                    # B's size -- 61952*512
                    size_2D = thing_2D.shape
                    thing_3D = np.zeros((size_3D[0], size_3D[1], size_3D[2]), dtype=np.complex64)
                    if index == 0:
                        for i in range(size_3D[0]):
                            thing_3D[i, :, :] = thing_2D[size_3D[1] * i:size_3D[1] * (i + 1), :]
                    elif index == 1:
                        for i in range(size_3D[1]):
                            thing_3D[:, i, :] = thing_2D[size_3D[0] * i:size_3D[0] * (i + 1), :]
                    elif index == 2:
                        for i in range(size_3D[2]):
                            thing_3D[:, :, i] = thing_2D[size_3D[0] * i:size_3D[0] * (i + 1), :]
                    else:
                        return -996

                    return thing_3D

                # -- 121*512*512
                # A_complex = torch.tensor(A_complex, dtype=torch.complex64).cuda().unsqueeze(0)
                A_new = unfoldTo2D(A_complex, 1)
                A_new = svd_zl(A_new,cut1)
                A_new = pileTo3D(A_new, 1)

                A_new = unfoldTo2D(A_new, 2)
                A_new = svd_zl(A_new,cut2)
                A_new = pileTo3D(A_new, 2)

                A_new = unfoldTo2D(A_new, 0)

                A_no = A[80 * 768:, :]
                # -- 61440*768
                svd_input = np.concatenate((A_new, A_no), 0)
                # -- 62001*768
                svd_input = torch.tensor(svd_input, dtype=torch.complex64)
                # =============================================== SVD
                A_svd = svd_zl(svd_input, cut0)

                # -- 62001*768
                # hankel回最初の情况
                A_svd = np.reshape(A_svd, [size_temp[0], size_temp[1], size_temp[2]], order='F')
                # -- 62001*64*12

                # 逆hankel, 除weight，再保真
                kcomplex_h = row2im(A_svd, size_data, ksize)

                # -- 256*256*8

                # 逆w和保真
                rec_Image = np.zeros((coil, 256, 256), dtype=np.complex64)
                k_complex = np.zeros((coil, 256, 256), dtype=np.complex64)
                k_complex2 = np.zeros((coil, 256, 256), dtype=np.complex64)
                for ii in range(coil):
                    def wgt2k(X, W, DC):
                        Y = np.multiply(X, 1. / W)
                        Y[W == 0] = DC[W == 0]
                        return Y
                    # weight逆，Ksample是欠采图
                    k_complex[ii, :, :] = wgt2k(kcomplex_h[:, :, ii], weight[ii, :, :], Ksample[ii, :, :])
                    # 保真
                    k_complex2[ii, :, :] = Ksample[ii, :, :] + k_complex[ii, :, :] * (1 - mask[ii, :, :])
                    rec_Image[ii, :, :] = np.fft.ifft2(k_complex2[ii, :, :])


                    ######乘weight
                k_w = np.zeros((coil, 256, 256), dtype=np.complex64)
                for i in range(coil):
                    k_w[i, :, :] = k2wgt(k_complex2[i, :, :], weight[i, :, :])
                #####再乘H得hankel矩阵
                k_w = k_w.transpose(1, 2, 0)
                hankel = im2row(k_w, ksize)
                size_temp = hankel.shape
                A = np.reshape(hankel, [size_temp[0], size_temp[1] * size_temp[2]],
                               order='F')  # max: 14.925017 (matlab:14.9250) (64009, 128)
                A_temp = np.zeros((80, 768, 768))
                ans_1 = np.array(A_temp, dtype=np.complex64)
                for i in range(80):  # diu 49
                    cut = A[768 * i:768 * (i + 1)]
                    ans_1[i, :, :] = cut
                # write_kdata(ans_1[0,:,:],'genxin_hankel')
                ans = np.concatenate((np.real(ans_1), np.imag(ans_1)), 0)  # 242,512,512 qiancai
                x_mean = torch.tensor(ans, dtype=torch.float32).cuda().unsqueeze(0)

                ##======================================================= Corrector
                x1, x2, x3, x_mean = corrector_update_fn(x1, x2, x3, x_mean, vec_t, model=model)
                # 尺寸恢复
                x_mean = x_mean.cpu().numpy().squeeze(0)  # (8,6,256,256)
                A_complex = x_mean[0:80, :, :] + 1j * x_mean[80:, :, :]

                # -- 121*512*512
                # A_complex = torch.tensor(A_complex, dtype=torch.complex64).cuda().unsqueeze(0)
                A_new_1 = unfoldTo2D(A_complex, 1)
                A_new_1 = svd_zl(A_new_1, cut1)
                A_new_1 = pileTo3D(A_new_1, 1)

                A_new_1 = unfoldTo2D(A_new_1, 2)
                A_new_1 = svd_zl(A_new_1, cut2)
                A_new_1 = pileTo3D(A_new_1, 2)

                A_new_1 = unfoldTo2D(A_new_1, 0)
                A_no = A[80 * 768:, :]

                # -- 61440*768
                svd_input = np.concatenate((A_new_1, A_no), 0)
                # -- 62001*768
                svd_input = torch.tensor(svd_input, dtype=torch.complex64)
                # =============================================== SVD
                A_svd = svd_zl(svd_input, cut0)

                # -- 62001*768
                # hankel回最初の情况
                A_svd = np.reshape(A_svd, [size_temp[0], size_temp[1], size_temp[2]], order='F')
                # -- 62001*64*12

                # 逆hankel, 除weight，再保真
                kcomplex_h = row2im(A_svd, size_data, ksize)

                # -- 256*256*8

                # 逆w和保真
                rec_Image = np.zeros((coil, 256, 256), dtype=np.complex64)
                k_complex = np.zeros((coil, 256, 256), dtype=np.complex64)
                k_complex2 = np.zeros((coil, 256, 256), dtype=np.complex64)
                for ii in range(coil):
                    def wgt2k(X, W, DC):
                        Y = np.multiply(X, 1. / W)
                        Y[W == 0] = DC[W == 0]
                        return Y

                    # weight逆，Ksample是欠采图
                    k_complex[ii, :, :] = wgt2k(kcomplex_h[:, :, ii], weight[ii, :, :], Ksample[ii, :, :])
                    # 保真
                    k_complex2[ii, :, :] = Ksample[ii, :, :] + k_complex[ii, :, :] * (1 - mask[ii, :, :])
                    rec_Image[ii, :, :] = np.fft.ifft2(k_complex2[ii, :, :])

                rec_Image_sos = np.sqrt(np.sum(np.square(np.abs(rec_Image)), axis=0))
                rec_Image_sos = rec_Image_sos / np.max(np.abs(rec_Image_sos))

                # PSNR
                psnr = compare_psnr(255 * abs(rec_Image_sos), 255 * abs(ori_data_sos), data_range=255)
                ssim = compare_ssim(abs(rec_Image_sos), abs(ori_data_sos), data_range=1)
                Isay = ' cut params:  ' + str(cut0) + '  ' + str(cut1) + '  ' + str(cut2)
                print(Isay + ' PSNR:', psnr, ' SSIM:', ssim)
                write_Data2(psnr, ssim, typezl)

                if max_ssim <= ssim:
                    max_ssim = ssim
                if max_psnr <= psnr:
                    max_psnr = psnr
                    write_Data('checkpoint', max_psnr, ssim, typezl)
                    write_images(abs(rec_Image_sos), osp.join('./result/' + typezl + 'bestRec' + '.png'))
                    io.savemat(osp.join('./result/' + typezl + 'bestRec.mat'), {'kncsn': rec_Image})

                k_w = np.zeros((coil, 256, 256), dtype=np.complex64)
                for i in range(coil):
                    k_w[i, :, :] = k2wgt(k_complex2[i, :, :], weight[i, :, :])
                #####再乘H得hankel矩阵
                k_w = k_w.transpose(1, 2, 0)
                hankel = im2row(k_w, ksize)
                size_temp = hankel.shape
                A = np.reshape(hankel, [size_temp[0], size_temp[1] * size_temp[2]],
                               order='F')  # max: 14.925017 (matlab:14.9250) (64009, 128)
                A_temp = np.zeros((80, 768, 768))
                ans_1 = np.array(A_temp, dtype=np.complex64)
                for i in range(80):  # diu 49
                    cut = A[768 * i:768 * (i + 1)]
                    ans_1[i, :, :] = cut
                # write_kdata(ans_1[0,:,:],'genxin_hankel')
                ans = np.concatenate((np.real(ans_1), np.imag(ans_1)), 0)  # 242,512,512 qiancai
                x_mean = torch.tensor(ans, dtype=torch.float32).cuda().unsqueeze(0)

            return x_mean  # inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

    return pc_sampler


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
    """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x)

            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
