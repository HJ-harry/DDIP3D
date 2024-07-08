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

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.linalg import orth
from pathlib import Path

from physics.ct import CT
from time import time
from utils import shrink, CG, clear, batchfy, _Dz, _DzT, PSNR, SSIM, update_ema, apply_ema_weights, restore_original_weights, \
    get_standard_score_openai_unet, get_standard_sde

# adaptation
from lora.lora import adapt_model, LoraInjectedConv1d, LoraInjectedConv2d, LoraInjectedLinear
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
        print('Run DDS 2D for CT reconstruction.',
            f'{self.args.T_sampling} sampling steps.',
            f'Task: {self.args.deg}.'
            f'Adaptation?: {self.adaptation}'
            )
        self.simplified_ddnm_plus(model)
            
            
    def simplified_ddnm_plus(self, model):
        args, config = self.args, self.config
        
        # parameters to be moved to args
        Nview = self.args.Nview
        rho = self.args.rho
        lamb = self.args.lamb
        n_ADMM = self.args.n_ADMM
        n_ADMM_adapt = self.args.n_ADMM_adapt
        n_CG = self.args.CG_iter

        if config.data.dataset == "phantom":
            root = None
        elif config.data.dataset == "AAPM":
            root = Path(config.data.root) / f"{config.data.vol_name}"
        print(f"Retrieving test data: {config.data.dataset}")
        # Root id
        root_id = Path(config.data_id.root)
        
        # Exceeding 3 causes GPU OOM issues
        self.max_gpu_mc = self.args.max_gpu_mc
        # iterated to accumulate gradients
        self.num_accumulation_round = self.args.num_mc // self.max_gpu_mc
        
        if self.args.num_test_slice != "all":
            self.args.num_test_slice = int(self.args.num_test_slice)
        
        # Specify save directory for saving generated samples
        in_dist_reg = True if self.args.in_dist_reg else False
        config_file = args.config.split('.')[0]
        save_root = Path(f'{self.args.save_root}/{config_file}/{self.args.deg}_view{self.args.Nview}/test_slice{self.args.num_test_slice}')
        save_root = save_root / f"adapt_{self.adaptation}_mc{self.args.num_mc}" \
            / f"lr{self.args.lr}_{self.args.num_steps}" \
            / f"n_ADMM{n_ADMM}" / f"n_CG{n_CG}" \
            / f"rho{self.args.rho}" / f"lamb{self.args.lamb}"
        save_root.mkdir(parents=True, exist_ok=True)

        irl_types = ['vol', 'input', 'recon', 'label', 'progress']
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
        
        # Read all ID data for regularization
        fname_list_id = sorted(os.listdir(root_id))
        x_orig_id = []
        for fname in fname_list_id:
            img = torch.from_numpy(np.load(os.path.join(root_id, fname)))
            h, w = img.shape
            img = img.view(1, 1, h, w)
            x_orig_id.append(img)
        x_orig_id = torch.cat(x_orig_id, dim=0)
        if self.args.num_test_slice != "all":
            half = self.args.num_test_slice // 2
            x_orig_id = x_orig_id[128-half:128+half, :, :, :]
        # total img count of id data
        n_id = x_orig_id.shape[0]
        
        img_shape = (x_orig.shape[0], config.data.channels, config.data.image_size, config.data.image_size)
        
        if self.args.deg == "SV-CT":
            A_funcs = CT(img_width=256, radon_view=self.args.Nview, uniform=True, circle=False, device=config.device)
        elif self.args.deg == "LA-CT":
            A_funcs = CT(img_width=256, radon_view=self.args.Nview, uniform=False, circle=False, device=config.device)
        
        A = lambda z: A_funcs.A(z)
        Ap = lambda z: A_funcs.A_dagger(z)
        
        x_orig = x_orig.to(self.device)
        y = A(x_orig)
        _, _, yh, yw = y.shape
        Apy = Ap(y)
        ATy = A_funcs.AT(y)
        
        if self.args.use_diffusionmbir:
            del_z = torch.zeros(img_shape, device=self.device)
            udel_z = torch.zeros(img_shape, device=self.device)
        
        def Acg(x):
            return A_funcs.AT(A_funcs.A(x))
        
        def Acg_TV(x):
            return A_funcs.AT(A_funcs.A(x)) + rho * _DzT(_Dz(x))
        
        def ADMM(x, ATy, n_ADMM=n_ADMM):
            nonlocal del_z, udel_z
            for _ in range(n_ADMM):
                bcg_TV = ATy + rho * (_DzT(del_z) - _DzT(udel_z))
                x = CG(Acg_TV, bcg_TV, x, n_inner=n_CG)
                del_z = shrink(_Dz(x) + udel_z, lamb / rho)
                udel_z = _Dz(x) - del_z + udel_z
            return x
        
        def ADMM_adapt(x, ATy, n_ADMM=n_ADMM_adapt):
            shape = (1, self.args.max_gpu_mc, 256, 256)
            del_z = torch.zeros(shape, device=self.device)
            udel_z = torch.zeros(shape, device=self.device)
            for _ in range(n_ADMM):
                bcg_TV = ATy + rho * (_DzT(del_z) - _DzT(udel_z))
                x = CG(Acg_TV, bcg_TV, x, n_inner=n_CG)
                del_z = shrink(_Dz(x) + udel_z, lamb / rho)
                udel_z = _Dz(x) - del_z + udel_z
            return x
        
        for idx in range(Apy.shape[0]):
            input = np.abs(clear(Apy[idx, ...]))
            plt.imsave(str(save_root / "input" / f"{str(idx).zfill(3)}.png"), input, cmap='gray')
            label = np.abs(clear(x_orig[idx, ...]))
            plt.imsave(str(save_root / "label" / f"{str(idx).zfill(3)}.png"), label, cmap='gray')
        
        """
        Actual inference running...
        """
        
        skip = config.diffusion.num_diffusion_timesteps // args.T_sampling
        
        times = range(0, 1000, skip)
        times_next = [-1] + list(times[:-1])
        times_pair = zip(reversed(times), reversed(times_next))
        
        init_time = (torch.ones(1) * times[-1])
        at = compute_alpha(self.betas.cpu(), init_time.long())
        
        # volume
        noise = torch.randn(
            x_orig.shape[0],
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
        )
        x = at.sqrt() * Apy.cpu() + (1 - at).sqrt() * noise
            
        n = x.size(0)
        x0_preds = []
        xs = [x]  # volume
        
        # reverse diffusion sampling
        for i, j in tqdm.tqdm(times_pair, total=len(times)):
            """
            Block 1: Adaptation with a single slice among the volume - in expectation
            """
            if args.adaptation:
                t = (torch.ones(1) * i).to("cuda")
                next_t = (torch.ones(1) * j).to("cuda")
                t_mod = int((t // 20).item())
                if (self.args.end_t <= t <= self.args.start_t) and (t_mod % self.args.adapt_every_k == 0):   
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    
                    print(f"Running adaptation at {i} / 1000")
                    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
                    for _ in range(args.num_steps):
                        optim.zero_grad()
                        for _ in range(self.num_accumulation_round):
                            # Monte carlo sample a random slice
                            samp_idx = random.sample(range(n - self.max_gpu_mc), 1)[0]
                            bcg = ATy[samp_idx:samp_idx+self.max_gpu_mc, ...].view(self.max_gpu_mc, 1, config.data.image_size, config.data.image_size)
                            y_idx = y[samp_idx:samp_idx+self.max_gpu_mc, ...].view(self.max_gpu_mc, 1, yh, yw)
                            xt = xs[-1][samp_idx:samp_idx+self.max_gpu_mc, ...].to(self.device).view(self.max_gpu_mc, 1, config.data.image_size, config.data.image_size)

                            # Use the sampled xt for adaptation. Doesn't have to be the same for every iter.
                            et = model(xt, t)[:, :1]
                            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                            # v2
                            x0_t = ADMM_adapt(x0_t, bcg, n_ADMM=n_ADMM_adapt)
                            # scale if num_acc_round > 1
                            loss = adapt_loss_fn(A(x0_t), y_idx) / self.num_accumulation_round
                            loss.backward()
                        optim.step()
            
            """
            Block 2: Inference in parallel
            """ 
            with torch.no_grad():
                t = (torch.ones(n) * i).to("cuda")
                next_t = (torch.ones(n) * j).to("cuda")
                at = compute_alpha(self.betas, t.long())
                at_next = compute_alpha(self.betas, next_t.long())
                xt = xs[-1].to('cuda')
                xt_batch = batchfy(xt, batch_size)
                et_agg = list()
                for _, xt_batch_sing in enumerate(xt_batch):
                    t = torch.ones(xt_batch_sing.shape[0], device=self.device) * i
                    et_sing = model(xt_batch_sing, t)
                    et_agg.append(et_sing)
                et = torch.cat(et_agg, dim=0)

                if et.size(1) == 2:
                    et = et[:, :1]

                # 1. Tweedie
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                # 2. Data consistency (ADMM TV)
                x0_t_hat = ADMM(x0_t, ATy, n_ADMM=n_ADMM)

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
        
        cnt = 0
        psnr_avg = 0
        ssim_avg = 0
        for idx in range(x.shape[0]):
            recon = clear(x[idx, ...])
            label = clear(x_orig[idx, ...])
            plt.imsave(str(save_root / "recon" / f"{str(idx).zfill(3)}.png"), recon, cmap='gray')
            plt.imsave(str(save_root / "label" / f"{str(idx).zfill(3)}.png"), label, cmap='gray')
            
            psnr = PSNR(recon, label)
            ssim = SSIM(recon, label, data_range=recon.max())
            
            psnr_avg += psnr
            ssim_avg += ssim
            cnt += 1
            
        summary = {}
        psnr_avg /= cnt
        ssim_avg /= cnt
        summary["results"] = {"PSNR": psnr_avg, "SSIM": ssim_avg}
        with open(str(save_root / f"summary.json"), 'w') as f:
            json.dump(summary, f)
            
        # Save LoRA weights for tuning 2d later on (TODO)
        adapted_lora_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, (LoraInjectedLinear, LoraInjectedConv2d, LoraInjectedConv1d)):
                adapted_lora_weights[name] = {pname: p.data.clone().to('cpu') for pname, p in module.named_parameters()}
        torch.save(adapted_lora_weights, str(save_root / "adapted_lora_weights.pth"))


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
