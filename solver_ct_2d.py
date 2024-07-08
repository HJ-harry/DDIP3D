import os
import logging
import time
import glob
import json
import sys
import odl

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.utils.data as data

from datasets import get_dataset

import torchvision.utils as tvu
import lpips

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, classifier_defaults, args_to_dict
from guided_diffusion.utils import get_alpha_schedule
import random

from scipy.linalg import orth
from pathlib import Path
from physics.ct import CT
from time import time
from utils import shrink, CG, clear, batchfy, _Dz, _DzT, PSNR, SSIM, \
    get_standard_score_openai_unet, get_standard_sde

# adaptation
from lora.lora import adapt_model
from lora.adaptation import adapt_loss_fn
from guided_diffusion.ema import ExponentialMovingAverage



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
    def __init__(self, args, config, device=None, coeff_schedule="ddnm"):
        self.args = args
        self.coeff_schedule = coeff_schedule
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
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        nparams = count_parameters(model)
        print(f"Number of parameters: {nparams}")
        ckpt = self.config.model.model_ckpt
        ckpt_f = torch.load(ckpt, map_location=self.device)
        if self.config.model.type == "openai":
            model.load_state_dict(ckpt_f)
        elif self.config.model.type == "scd-unet":
            ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
            ema.load_state_dict(ckpt_f)
            ema.copy_to(model.parameters())
            del ema
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
        print('Running DDIP for CT reconstruction.',
            f'{self.args.T_sampling} sampling steps.',
            f'Task: {self.args.deg}.'
            f'Adaptation?: {self.adaptation}'
            )
        self.simplified_ddnm_plus(model)
            
            
    def simplified_ddnm_plus(self, model):
        args, config = self.args, self.config

        if config.data.dataset == "phantom":
            root = None
        elif config.data.dataset == "AAPM":
            root = Path(config.data.root) / f"{config.data.vol_name}"
        print(f"Retrieving test data: {config.data.dataset}")
        
        # parameters to be moved to args
        Nview = self.args.Nview
        rho = self.args.rho
        lamb = self.args.lamb
        n_ADMM = 1
        n_CG = self.args.CG_iter
        
        if self.args.num_test_slice != "all":
            self.args.num_test_slice = int(self.args.num_test_slice)
        
        # Specify save directory for saving generated samples
        config_file = args.config.split('.')[0]
        save_root = Path(f'{self.args.save_root}/{config_file}/{self.args.deg}_view{self.args.Nview}')
        save_root = save_root / f"adapt_{self.adaptation}" \
            / f"lr{self.args.lr}_{self.args.num_steps}" \
            / f"lora_rank{self.args.lora_rank}"
        save_root.mkdir(parents=True, exist_ok=True)

        irl_types = ['input', 'recon', 'label', 'progress']
        for t in irl_types:
            save_root_f = save_root / t
            save_root_f.mkdir(parents=True, exist_ok=True)
        
        # read all data

        batch_size = self.args.batch_size
        print("Loading all data")
        fname_list = os.listdir(root)
        fname_list = sorted(fname_list, key=lambda x: float(x.split(".")[0]))
        x_orig = []
        for fname in fname_list:
            just_name = fname.split('.')[0]
            img = torch.from_numpy(np.load(os.path.join(root, fname), allow_pickle=True))
            h, w = img.shape
            img = img.view(1, 1, h, w)
            x_orig.append(img)
        x_orig = torch.cat(x_orig, dim=0)
        if self.args.num_test_slice != "all":
            half = self.args.num_test_slice // 2
            x_orig = x_orig[128-half:128+half, :, :, :]
        print(f"Data loaded shape : {x_orig.shape}")
        
        img_shape = (x_orig.shape[0], config.data.channels, config.data.image_size, config.data.image_size)
        
        if self.args.deg == "SV-CT":
            A_funcs = CT(img_width=256, radon_view=self.args.Nview, uniform=True, circle=False, device=config.device)
        elif self.args.deg == "LA-CT":
            A_funcs = CT(img_width=256, radon_view=self.args.Nview, uniform=False, circle=False, device=config.device)
        
        A = lambda z: A_funcs.A(z)
        Ap = lambda z: A_funcs.A_dagger(z)
        
        x_orig = x_orig.to(self.device)
        y = A(x_orig)
        Apy = Ap(y)
        
        ATy = A_funcs.AT(y)
        
        if self.args.sampler == "DDS" or self.args.sampler == "DDNM":
            def Acg(x):
                return A_funcs.AT(A_funcs.A(x))
        elif self.args.sampler == "DiffPIR":
            def Acg(x, gamma=self.args.gamma):
                return x + gamma * A_funcs.AT(A_funcs.A(x))
        
        for idx in range(Apy.shape[0]):
            input = np.abs(clear(Apy[idx, ...]))
            plt.imsave(str(save_root / "input" / f"{str(idx).zfill(3)}.png"), input, cmap='gray')
            label = np.abs(clear(x_orig[idx, ...]))
            plt.imsave(str(save_root / "label" / f"{str(idx).zfill(3)}.png"), label, cmap='gray')
            
        """
        Actual inference running...
        """
        cnt = 0
        psnr_avg = 0
        ssim_avg = 0
        for idx in range(x_orig.shape[0]):
            print(f"{idx+1}/{x_orig.shape[0]} inference running")
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
                bcg = ATy[idx:idx+1, ...]
                y_idx = y[idx:idx+1, ...]
                """
                Block 1: Adaptation
                """
                if args.adaptation:
                    xt = xs[-1].to('cuda')
                    print(f"Running adaptation at {i} / 1000")
                    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
                    for _ in range(args.num_steps):
                        optim.zero_grad()
                        et = model(xt, t)[:, :1]
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                        x0_t = CG(Acg, bcg, x0_t, n_inner=self.args.CG_iter_adapt)
                        loss = adapt_loss_fn(A(x0_t), y_idx)
                        loss.backward()
                        optim.step()
                
                """
                Block 2: Inference after adaptation
                """
                with torch.no_grad():
                    xt = xs[-1].to('cuda')
                    
                    et = model(xt, t)

                    if et.size(1) == 2:
                        et = et[:, :1]

                    # 1. Tweedie
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                    # 2. Data consistency (CG)
                    x0_t_hat = CG(Acg, bcg, x0_t, n_inner=self.args.CG_iter)

                    eta = self.args.eta
                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                    # DDIM sampling
                    if j != 0:
                        xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et
                    # Final step
                    else:
                        xt_next = x0_t_hat

                    x0_preds.append(x0_t.to('cpu'))
                    xs.append(xt_next.to('cpu'))
                x = xs[-1]
            recon = clear(x)
            label = clear(x_orig[idx])
            
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
