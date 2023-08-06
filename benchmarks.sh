#!/bin/bash

#===================== Same Procedure on Images =====================
#  python  main.py --mode styleImages --lr 0.001 --device cuda:1\
#  --prefix Benchmark --loss l1 --batch_size 32 --burnin 10 \
#  --pretrained_style None\
#  --pretrained None\
#  --normalization instance

#===================== MultiSegment Style =====================
# python  main.py --mode styleMulti --lr 0.001 --device cuda:0\
#  --prefix Benchmark --test True --num_epochs 5 --test True\
#  --normalization instance --batch_size 16\
#  --pretrained_style /home/anna/style_results/HighLR2021-09-13_styleMulti/

#===================== Linear Style =====================
# ---------- Style ----------
# python  main.py --mode styleLinear --lr 0.001 --device cuda:0\
#  --prefix Benchmark --num_epochs 1 --test True\
#  --normalization instance --batch_size 16\
#  --pretrained_style /home/anna/style_results/deeplatent2021-06-09_chunk-1/

# ---------- Sides ----------
# python  main.py --mode sidesAE --lr 0.001 --device cuda:0 --test True\
#  --prefix Benchmark_ --loss l1 --num_epochs 100\
#  --pretrained_style /home/anna/style_results/deeplatent2021-06-09_chunk-1/\
#  --pretrained /home/anna/style_results/adv2021-06-21_chunk-1_sides/\
#  --batch_size 4 --burnin 1000


# python  main.py --mode sidesAE --lr 0.001 --device cuda:0 \
#  --prefix Benchmark_noStyleC1R1_ --loss l1 --num_epochs 1\
#  --pretrained_style None\
#  --pretrained None\
#  --tgt_dir ./MIAreview/Results/\
#  --file_in /itet-stor/klanna/bmicdatasets_bmicnas01/Sharing/klanna/datasets/sdan/data_19Nov2022/test_syn\
#  --target /itet-stor/klanna/bmicdatasets_bmicnas01/Sharing/klanna/datasets/sdan/data_19Nov2022/test_real\
#  --batch_size 4 --burnin 1000\
#  --weight_center 1.0\
#  --weight_real 1.0


path=/home/anna.susmelj@ad.biognosys.ch/MIDL/
python  pl_main.py --mode sidesAE --lr 0.001 --device cuda:0\
 --test 0\
 --prefix Ablation_woStyleL1Real_\
 --loss l1 --num_epochs 1\
 --pretrained_style None\
 --pretrained None\
 --burnin 50\
 --tgt_dir ./MIAreview/Results/\
 --file_in /itet-stor/klanna/bmicdatasets_bmicnas01/Sharing/klanna/datasets/sdan/data_19Nov2022/test_syn\
 --target /itet-stor/klanna/bmicdatasets_bmicnas01/Sharing/klanna/datasets/sdan/data_19Nov2022/test_real\
 --batch_size 2 

# path=/home/anna.susmelj@ad.biognosys.ch/MIDL/
#  python  pl_main.py --mode sidesAE --lr 0.001 --device cuda:0\
#  --test 0\
#  --prefix Benchmark_ --loss l1 --num_epochs 100\
#  --target $path/data/arm.h5\
#  --file_in $path/data/sigmat_multisegment_simulated.h5\
#  --pretrained_style None\
#  --pretrained /home/anna.susmelj@ad.biognosys.ch/MIDL/results/woStyleL22022-02-03_sidesAE/epoch=504-step=10604.ckpt\
#  --prefix woStyleL2Real\
#  --burnin 50\
#  --tgt_dir $path/results/\
#  --batch_size 32


# path=/home/anna.susmelj@ad.biognosys.ch/MIDL/
#  python  pl_main.py --mode sidesAE --lr 0.001 --device cuda:0\
#  --test 0\
#  --prefix Benchmark_ --loss l1 --num_epochs 100\
#  --target $path/data/arm.h5\
#  --file_in $path/data/sigmat_multisegment_simulated.h5\
#  --pretrained_style None\
#  --pretrained /home/anna.susmelj@ad.biognosys.ch/MIDL/results/woStyleL22022-02-03_sidesAE/epoch=504-step=10604.ckpt\
#  --prefix woStyleL2Real\
#  --burnin 50\
#  --tgt_dir $path/results/\
#  --batch_size 448


# path=/home/anna.susmelj@ad.biognosys.ch/MIDL/
#  python  pl_main.py --mode sidesAE --lr 0.001 --device cuda:0\
#  --test 0\
#  --prefix Benchmark_ --loss l2 --num_epochs 500\
#  --target $path/data/sigmat_multisegment_simulated.h5\
#  --file_in $path/data/sigmat_multisegment_simulated.h5\
#  --pretrained_style None\
#  --pretrained None\
#  --prefix woStyleL2OnlySyn\
#  --burnin 50\
#  --tgt_dir $path/results/\
#  --batch_size 352

# path=/home/anna.susmelj@ad.biognosys.ch/MIDL/
#  python  pl_main.py --mode sidesAE --lr 0.001 --device cuda:0\
#  --test 0\
#  --prefix Benchmark_ --loss l1 --num_epochs 500\
#  --target $path/data/sigmat_multisegment_simulated.h5\
#  --file_in $path/data/sigmat_multisegment_simulated.h5\
#  --pretrained_style None\
#  --pretrained /home/anna.susmelj@ad.biognosys.ch/MIDL/results/woStyleL2OnlySyn2022-02-07_sidesAE/epoch=419-step=6299.ckpt\
#  --prefix woStyleL2L1OnlySyn\
#  --burnin 50\
#  --tgt_dir $path/results/\
#  --batch_size 352

#===================== Simple Benchmark =====================
# python  main.py --mode sidesTwo --lr 0.001 --device cuda:1\
#  --prefix Benchmark --split left --test True --num_epochs 10\
#  --normalization batch --batch_size 32\
#  --pretrained_style /home/anna/style_results/deeplatent2021-06-09_chunk-1/\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/

# python  main.py --mode sidesTwo --lr 0.001 --device cuda:1\
#  --prefix Benchmark --split right --test True --num_epochs 10\
#  --normalization batch --batch_size 32\
#  --pretrained_style /home/anna/style_results/deeplatent2021-06-09_chunk-1/\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/


