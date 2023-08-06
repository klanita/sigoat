#!/bin/bash

#SBATCH  --output=./LOGS/%j.out
#SBATCH  --error=./LOGS/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=80G

source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
conda activate pytcu11

python  main.py --mode sidesAE --lr 0.001 --device cuda:0 \
 --prefix Benchmark_noStyleC1R1_ --loss l1 --num_epochs 100\
 --pretrained_style None\
 --pretrained None\
 --tgt_dir ./MIAreview/Results/\
 --file_in /itet-stor/klanna/bmicdatasets_bmicnas01/Sharing/klanna/datasets/sdan/data_19Nov2022/old_syn\
 --target /itet-stor/klanna/bmicdatasets_bmicnas01/Sharing/klanna/datasets/sdan/data_19Nov2022/real\
 --batch_size 128 --burnin 1000\
 --weight_center 1.0\
 --weight_real 1.0