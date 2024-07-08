import os
import logging
import time
import glob
import json
import sys
import odl
import functools

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import torch.utils.data as data

from datasets import get_dataset

import torchvision.utils as tvu
import lpips

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, classifier_defaults, args_to_dict
from guided_diffusion.utils import get_alpha_schedule
import random

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.linalg import orth
from pathlib import Path

from physics.mri import MulticoilMRI, SinglecoilMRI_comp
from time import time
from utils import shrink, CG, clear, batchfy, _Dz, _DzT, get_mask, real_to_nchw_comp, comp_to_nchw_real, PSNR, SSIM

# adaptation
from lora.lora import adapt_model, LoraInjectedConv1d, LoraInjectedConv2d, LoraInjectedLinear
from lora.adaptation import adapt_loss_fn



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        config_dict = vars(self.config.model)
        model = create_model(**config_dict)
        ckpt = self.config.model.model_ckpt
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"Model ckpt loaded from {ckpt}")
        model.to(self.device)
        model.convert_to_fp32()
        model.dtype = torch.float32
        model.eval()
        # Augment with adaptation parameters
        if self.args.adaptation:
            adapt_kwargs = {'r': int(self.args.lora_rank)}
            adapt_model(model, adapt_kwargs=adapt_kwargs)

        self.adaptation = True if self.args.adaptation else False
        print('Run DDS 2D for MRI reconstruction.',
            f'{self.args.T_sampling} sampling steps. ',
            f'Task: {self.args.deg}. '
            f'Adaptation?: {self.adaptation}'
            )
        self.simplified_ddnm_plus(model)
            
            
    def simplified_ddnm_plus(self, model):
        args, config = self.args, self.config
        img_size = config.data.image_size

        root_list = []
        if config.data.vol_name != "all":
            root = Path(config.data.root) / f"{config.data.vol_name}"
            root_list.append(root)
        else:
            root = Path(config.data.root)
            vol_names = os.listdir(root)
            for vol in vol_names:
                root_list.append((root / vol))
            
        print(f"Retrieving test data: {config.data.dataset}")
        
        # Iterate over all vols
        for root in root_list:
            vol = str(root).split('/')[-1]
            print(f"root: {root}")
            print(f"vol: {vol}")
        
            # Specify save directory for saving generated samples
            save_root = Path(f'{self.args.save_root}/{self.config.data.dataset}/{vol}/{self.args.mask_type}_acc{self.args.acc_factor}/cg_gamma{self.args.gamma}')
            save_root = save_root / f"adapt_{self.adaptation}" / f"lr{self.args.lr}_{self.args.num_steps}" / f"{self.args.start_t}_{self.args.end_t}_every{self.args.adapt_every_k}"
            save_root.mkdir(parents=True, exist_ok=True)

            irl_types = ['input', 'recon', 'label', 'progress']
            for t in irl_types:
                save_root_f = save_root / t
                save_root_f.mkdir(parents=True, exist_ok=True)
            
            # read all data: TODO
            print("Loading all data")
            root_img = root / "slice"
            root_mps = root / "mps"
            fname_list = sorted(os.listdir(root_img))
            x_orig = []
            mps_orig = []
            for fname in fname_list:
                img = torch.from_numpy(np.load(os.path.join(root_img, fname)))
                mps = torch.from_numpy(np.load(os.path.join(root_mps, fname)))
                h, w = img.shape
                c, h, w = mps.shape
                img = img.view(1, 1, h, w)
                mps = mps.view(1, c, h, w)
                x_orig.append(img)
                mps_orig.append(mps)
            x_orig = torch.cat(x_orig, dim=0)
            mps_orig = torch.cat(mps_orig, dim=0)
            print(f"Data loaded shape - img: {x_orig.shape}")
            print(f"                    mps: {mps_orig.shape}")
            
            img_shape = (x_orig.shape[0], config.data.channels, img_size, img_size)
            
            # MRI forward operator
            mask = get_mask(
                torch.zeros([1, 1, img_size, img_size]), 
                img_size,
                1,
                type=self.args.mask_type,
                acc_factor=self.args.acc_factor, 
                center_fraction=self.args.center_fraction,
            ).to("cuda")
            A_funcs = MulticoilMRI(mask=mask)
            
            # Alias
            A = lambda z, mps: A_funcs._A(z, mps)
            AT = lambda z, mps: A_funcs._AT(z, mps)
            Ap = lambda z, mps: A_funcs._Adagger(z, mps)
            
            def Acg(x, mps, gamma):
                return x + gamma * A_funcs._AT(A_funcs._A(x, mps), mps)
            
            y = torch.zeros_like(mps_orig)
            ATy = torch.zeros_like(x_orig)
            for idx in range(x_orig.shape[0]):
                x_idx = x_orig[idx:idx+1, ...].to(self.device)
                mps_idx = mps_orig[idx:idx+1, ...].to(self.device)
                y_idx = A(x_idx, mps_idx)
                y += torch.randn_like(y) * self.args.sigma_y
                ATy_idx = AT(y_idx, mps_idx)
                y[idx, ...] = y_idx
                ATy[idx, ...] = ATy_idx
                input = np.abs(clear(ATy_idx))
                label = np.abs(clear(x_idx))
                plt.imsave(str(save_root / "input" / f"{str(idx).zfill(3)}.png"), input, cmap='gray')
                plt.imsave(str(save_root / "label" / f"{str(idx).zfill(3)}.png"), label, cmap='gray')
                
            """
            Actual inference running...
            """
            cnt = 0
            psnr_avg = 0
            ssim_avg = 0
            for idx in range(x_orig.shape[0]):
                x = torch.randn_like(x_orig[idx:idx+1, ...]).to(self.device)
                skip = config.diffusion.num_diffusion_timesteps//args.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]
                
                # generate time schedule
                times = range(0, 1000, skip)
                times_next = [-1] + list(times[:-1])
                times_pair = zip(reversed(times), reversed(times_next))
                
                # reverse diffusion sampling
                for i, j in tqdm.tqdm(times_pair, total=len(times)):
                    t = (torch.ones(n) * i).to("cuda")
                    next_t = (torch.ones(n) * j).to("cuda")
                    
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    ATy_idx = ATy[idx:idx+1, ...].to(self.device)
                    y_idx = y[idx:idx+1, ...].to(self.device)
                    mps_idx = mps_orig[idx:idx+1, ...].to(self.device)
                    Acg_idx = functools.partial(Acg, mps=mps_idx, gamma=self.args.gamma)
                    """
                    Block 1: Adaptation
                    """
                    if args.adaptation:
                        xt = xs[-1].to('cuda')
                        # [1, 1, 240, 240] comp -> [1, 2, 240, 240] real
                        xt = comp_to_nchw_real(xt)
                        print(f"Running adaptation at {i} / 1000")
                        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
                        for _ in range(args.num_steps):
                            optim.zero_grad()
                            et = model(xt, t)[:, :2]
                            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                            # [1, 2, 240, 240] real -> [1, 1, 240, 240] comp
                            x0_t = real_to_nchw_comp(x0_t)
                            bcg = x0_t + self.args.gamma * ATy_idx
                            x0_t = CG(Acg_idx, bcg, x0_t, n_inner=1)
                            loss = adapt_loss_fn(A(x0_t, mps_idx), y_idx)
                            loss.backward()
                            optim.step()
                            
                    """
                    Block 2: Inference after adaptation
                    """
                    with torch.no_grad():
                        xt = xs[-1].to('cuda')
                        # [1, 1, 240, 240] comp -> [1, 2, 240, 240] real
                        xt = comp_to_nchw_real(xt)
                        et = model(xt, t)[:, :2]

                        # 1. Tweedie
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                        # [1, 2, 240, 240] real -> [1, 1, 240, 240] comp
                        x0_t = real_to_nchw_comp(x0_t)
                        # 2. Data consistency (CG)
                        bcg = x0_t + self.args.gamma * ATy_idx
                        x0_t = CG(Acg_idx, bcg, x0_t, n_inner=5)

                        eta = self.args.eta
                        c1 = (1 - at_next).sqrt() * eta
                        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                        # DDIM sampling
                        if j != 0:
                            # [1, 1, 240, 240] comp -> [1, 2, 240, 240] real
                            et = real_to_nchw_comp(et)
                            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
                        # Final step
                        else:
                            xt_next = x0_t

                        x0_preds.append(x0_t.to('cpu'))
                        xs.append(xt_next.to('cpu'))
                    x = xs[-1]
                recon = np.abs(clear(x))
                label = np.abs(clear(x_orig[idx]))
                
                psnr = PSNR(recon, label)
                ssim = SSIM(recon, label, data_range=recon.max())
                
                psnr_avg += psnr
                ssim_avg += ssim
                cnt += 1
                
                plt.imsave(str(save_root / "recon" / f"{str(idx).zfill(3)}.png"), recon, cmap='gray')
            summary = {}
            psnr_avg /= cnt
            ssim_avg /= cnt
            summary["results"] = {"PSNR": psnr_avg, "SSIM": ssim_avg}
            with open(str(save_root / f"summary.json"), 'w') as f:
                json.dump(summary, f)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
