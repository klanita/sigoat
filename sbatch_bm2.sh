#!/bin/bash

#SBATCH  --output=./LOGS/%j.out
#SBATCH  --error=./LOGS/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G
###SBATCH  --account=staff

source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
conda activate pytcu11

tgtpath=/itet-stor/klanna/bmicdatasets_bmicnas01/Sharing/klanna
path=/home/anna.susmelj@ad.biognosys.ch/MIDL/
python -u pl_main.py --mode sidesAE --lr 0.005 --device cuda:0\
 --test 0\
 --prefix Ablation_woStyleL2L1_wocenter_\
 --loss l2 --num_epochs 600\
 --pretrained_style None\
 --pretrained None\
 --burnin 0\
 --tgt_dir $tgtpath/results/SDAN\
 --file_in $tgtpath/datasets/sdan/data_19Nov2022/old_syn\
 --target $tgtpath/datasets/sdan/data_19Nov2022/test_real\
 --batch_size 256 

#  --burnin 200\
# Ablation_woStyleL2L1_
# ckpt=$tgtpath/results/SDAN/Ablation_woStyleL2L1Real_batch_2023-08-07_sidesAE/epoch\=248-step\=5229.ckpt

# echo $ckpt

# python -u pl_main.py --mode sidesAE --lr 0.005 --device cuda:0\
#  --test 0\
#  --prefix Ablation_woStyleL2L1_\
#  --loss l1 --num_epochs 600\
#  --pretrained_style None\
#  --pretrained $ckpt\
#  --burnin 200\
#  --tgt_dir $tgtpath/results/SDAN\
#  --file_in $tgtpath/datasets/sdan/data_19Nov2022/old_syn\
#  --target $tgtpath/datasets/sdan/data_19Nov2022/test_real\
#  --batch_size 256 


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