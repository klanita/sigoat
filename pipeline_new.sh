#!/bin/bash

python  main.py\
 --file_in /home/anna/data_19Nov2022/old_syn\
 --target /home/anna/data_19Nov2022/real\
 --mode styleLinear\
 --lr 0.00001 --device cuda:1\
 --prefix MIA \
 --num_epochs 5 \
 --burnin 10 \
 --normalization instance \
 --batch_size 16 \
 --weight_sides 10\
 --weight_adv_latent 0.01 \
 --weight_cycle 0.001 \
 --n_iters 4 \
 --loss l1\
 --pretrained_style /home/anna/ResultsSignalDA/Benchmark2021-09-30_styleLinear/ \
 --pretrained  None
 #  /home/anna/style_results/ReproduceNotPretrained2021-09-22_styleLinear_instance/\
