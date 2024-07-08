#!/bin/bash

T_sampling=50
eta=0.85
start_t=980
end_t=20
num_mc=3
lr='1e-3'
num_steps=5

python main.py \
    --config FASTMRI_BRAIN2KNEE.yml \
    --deg MRI \
    --T_sampling $T_sampling \
    --eta ${eta} \
    --sigma_y 0.01 \
    --save_root ./results/mri_3d \
    --mask_type 'uniform1d' \
    --acc_factor 4 \
    --gamma 5.0 \
    --adaptation \
    --start_t ${start_t} \
    --end_t ${end_t} \
    --num_mc ${num_mc} \
    --lr ${lr} \
    --num_steps ${num_steps}