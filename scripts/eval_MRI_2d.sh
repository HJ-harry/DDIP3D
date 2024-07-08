#!/bin/bash

T_sampling=50
eta=0.85
start_t=980
end_t=20

lr='1e-3'
num_steps=5

python main.py \
    --config FASTMRI_BRAIN2KNEE.yml \
    --adaptation \
    --deg MRI \
    --T_sampling $T_sampling \
    --eta ${eta} \
    --save_root ./results/mri_2d \
    --mask_type 'uniform1d' \
    --acc_factor 4 \
    --gamma 5.0 \
    --use_2d \
    --start_t ${start_t} \
    --end_t ${end_t} \
    --lr ${lr} \
    --num_steps ${num_steps}
    