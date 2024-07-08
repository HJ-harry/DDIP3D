import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

from pathlib import Path
from datetime import datetime

from solver_ct_3d import Diffusion as Diffusion_CT
from solver_ct_3d_diffmbir import Diffusion as Diffusion_CT_diffmbir
from solver_ct_2d import Diffusion as Diffusion_CT_2d
from solver_mri_3d import Diffusion as Diffusion_MRI
from solver_mri_2d import Diffusion as Diffusion_MRI_2d

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--sampler", type=str, default="DDS", choices=["DDS", "DPS", "DDNM", "DiffPIR"])
    parser.add_argument("--n_ADMM", type=int, default=5, help="Outer ADMM loop")
    parser.add_argument("--n_ADMM_adapt", type=int, default=5, help="Inner ADMM loop for adaptation")
    parser.add_argument("--CG_iter", type=int, default=5, help="Inner number of iterations for CG")
    parser.add_argument("--CG_iter_adapt", type=int, default=5, help="CG loop for adaptation.")
    parser.add_argument("--Nview", type=int, default=16, help="number of projections for CT")
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument("--deg", type=str, required=True, help="Degradation")
    parser.add_argument("--sigma_y", type=float, default=0., help="sigma_y")
    parser.add_argument("--eta", type=float, default=0.85, help="Eta")
    parser.add_argument("--gamma", type=float, default=5.0, help="Weighting on the proximal term when running CG")
    parser.add_argument("--T_sampling", type=int, default=50, help="Total number of sampling steps")    
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--save_root", type=str, default="./results", help="Where the results will be saved")
    parser.add_argument("--batch_size", type=int, default=12, help="Mini-batch size that will be run in parallel for sampling")
    # exp setting: 2d, 3d, finetune, id, etc.
    parser.add_argument("--test_run", action="store_true",
        help="Inference on 256x256x8 rather than 256x256x256 to reduce compute",
    )
    parser.add_argument("--num_test_slice", type=str, default="all",
        help="Number of testing slices. If set to 'all', run on total volume.",
    )
    parser.add_argument("--use_2d", action="store_true", help="2D DDS inference to see the performance")
    parser.add_argument("--use_2d_finetune", action="store_true", 
                        help="Start from 3D LoRA adapted weights, fine-tune on 2D slice")
    parser.add_argument("--adapt_ckpt_3d", type=str, help="3D LoRA weights path")
    # adapt
    parser.add_argument("--adaptation", action="store_true")
    parser.add_argument("--max_gpu_mc", type=int, default=3, help="Max batch size for monte carlo sample per GPU.")
    parser.add_argument("--num_mc", type=int, default=1, help="Monte carlo sample for adaptation")
    parser.add_argument('--start_t', type=int, default=1000)
    parser.add_argument('--end_t', type=int, default=0)
    parser.add_argument('--adapt_every_k', type=int, default=1)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for adaptation')
    parser.add_argument('--num_steps', default=3, type=int, help='num. of optimization steps taken per sampl. step')
    parser.add_argument('--adapt_freq', default=1, help='freq. of adaptation step in sampl.')
    parser.add_argument('--lora_rank', default=4, help='lora kwargs impl. of rank')
    # MRI
    parser.add_argument("--mask_type", type=str, default="poisson", help="Undersampling type")
    parser.add_argument("--acc_factor", type=int, default=15, help="acceleration factor")
    parser.add_argument("--center_fraction", type=float, default=0.08, help="ACS region. Not used for 2D sampling types")
    # Initialization strategy
    parser.add_argument('--init', default='noise', type=str, choices=['noise', 'y_same_noise', 'y_diff_noise'],
                        help='Weighting for in-dist denoising loss.')
    # DiffusionMBIR
    parser.add_argument("--use_diffusionmbir", action="store_true")
    parser.add_argument("--rho", type=float, default=1.0, help="rho")
    parser.add_argument("--lamb", type=float, default=0.4, help="lambda for TV")
    # Meta learning
    parser.add_argument('--use_meta', action='store_true', 
                        help="Use Reptile meta learning algorithm for better initialization(?) For further finetuning on 2D")
    parser.add_argument('--meta_step_size', default=0.1, type=float, 
                        help='Step size for outer Reptile. 1.0 reduces to standard 3D training.')
    parser.add_argument('--meta_linear_decay', action='store_true', 
                        help='Use linear decay as proposed in Reptile')
    
    
    args = parser.parse_args()
    # parse config file
    with open(os.path.join("configs/vp", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    try:
        if "CT" in args.deg:
            if args.use_2d:
                runner = Diffusion_CT_2d(args, config)
            elif args.use_diffusionmbir:
                runner = Diffusion_CT_diffmbir(args, config)
            else:
                runner = Diffusion_CT(args, config)
        elif "MRI" in args.deg:
            if args.use_2d:
                runner = Diffusion_MRI_2d(args, config)
            else:
                if args.use_diffusionmbir:
                    runner = Diffusion_MRI_type2_diffmbir(args, config)
                else:
                    runner = Diffusion_MRI(args, config)
        else:
            print(f"Got {args.deg} as degradation! Expected one of 'CT' or 'MRI'.")
        runner.sample()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
