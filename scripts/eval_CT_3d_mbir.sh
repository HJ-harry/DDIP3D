#!/bin/bash

Nview=60
T_sampling=50
eta=0.85
start_t=980
end_t=20

num_test_slice=8

lr='1e-3'
num_steps=10
init='y_diff_noise'
lora_rank=4
num_mc=3
max_gpu_mc=3
n_ADMM=5
CG_iter=5

rho=0.5
lamb=0.01
n_ADMM_adapt=1

# Adapt version
python main.py \
    --config ELLIPSES2AAPM.yml \
    --adaptation \
    --Nview $Nview \
    --eta $eta \
    --T_sampling $T_sampling \
    --deg "SV-CT" \
    --save_root ./results/ct_3d_mbir \
    --start_t ${start_t} \
    --end_t ${end_t} \
    --num_mc ${num_mc} \
    --max_gpu_mc ${max_gpu_mc} \
    --lr ${lr} \
    --num_steps ${num_steps} \
    --ema_alpha ${ema_alpha} \
    --num_test_slice ${num_test_slice} \
    --init ${init} \
    --lora_rank ${lora_rank} \
    --use_diffusionmbir \
    --n_ADMM ${n_ADMM} \
    --n_ADMM_adapt ${n_ADMM_adapt} \
    --CG_iter ${CG_iter} \
    --rho ${rho} \
    --lamb ${lamb}