#!/bin/bash

#SBATCH  --output=./LOGS/%j.out
#SBATCH  --error=./LOGS/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G
###SBATCH  --account=staff

source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
conda activate pytcu11

path=/home/anna.susmelj@ad.biognosys.ch/MIDL/
python -u pl_main.py --mode sidesAE --lr 0.005 --device cuda:0\
 --test 0\
 --prefix Ablation_woStyleL2L1Real_batch_\
 --loss l2 --num_epochs 300\
 --pretrained_style None\
 --pretrained None\
 --burnin 200\
 --tgt_dir /itet-stor/klanna/bmicdatasets_bmicnas01/Sharing/klanna/results/SDAN\
 --file_in /itet-stor/klanna/bmicdatasets_bmicnas01/Sharing/klanna/datasets/sdan/data_19Nov2022/old_syn\
 --target /itet-stor/klanna/bmicdatasets_bmicnas01/Sharing/klanna/datasets/sdan/data_19Nov2022/test_real\
 --batch_size 256 

# python  main.py --mode sidesAE --lr 0.001 --device cuda:0 \
#  --prefix Benchmark_noStyleC1R1_ --loss l1 --num_epochs 100\
#  --pretrained_style None\
#  --pretrained None\
#  --tgt_dir ./MIAreview/Results/\
#  --file_in /itet-stor/klanna/bmicdatasets_bmicnas01/Sharing/klanna/datasets/sdan/data_19Nov2022/old_syn\
#  --target /itet-stor/klanna/bmicdatasets_bmicnas01/Sharing/klanna/datasets/sdan/data_19Nov2022/real\
#  --batch_size 32 --burnin 1000\
#  --weight_center 1.0\
#  --weight_real 0.0