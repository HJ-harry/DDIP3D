#!/bin/bash

Nview=60
T_sampling=50
eta=0.85
start_t=980
end_t=20
gamma=5.0
lr='1e-3'
num_steps=10
gamma=5.0

num_test_slice=8
lora_rank=4

python main.py \
    --config ELLIPSES2AAPM.yml \
    --use_2d \
    --adaptation \
    --Nview ${Nview} \
    --eta ${eta} \
    --T_sampling ${T_sampling} \
    --deg "SV-CT" \
    --save_root ./results/ct_2d \
    --num_test_slice ${num_test_slice} \
    --lora_rank ${lora_rank} \
    --lr ${lr} \
    --num_steps ${num_steps} \
    --gamma ${gamma}