#!/bin/bash

Nview=60
T_sampling=50
eta=0.85
start_t=980
end_t=20

num_test_slice='all'

lr='1e-3'
num_steps=10
ema_alpha=0.0
init='y_diff_noise'
lora_rank=4
num_mc=3

python main.py \
    --config ELLIPSES2AAPM.yml \
    --adaptation \
    --Nview $Nview \
    --eta $eta \
    --T_sampling $T_sampling \
    --deg "SV-CT" \
    --sigma_y 0.0 \
    --save_root ./results/ct_3d \
    --start_t ${start_t} \
    --end_t ${end_t} \
    --num_mc ${num_mc} \
    --lr ${lr} \
    --num_steps ${num_steps} \
    --ema_alpha ${ema_alpha} \
    --num_test_slice ${num_test_slice} \
    --init ${init} \
    --lora_rank ${lora_rank}
