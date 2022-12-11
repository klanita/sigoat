#!/bin/bash

# step 1 ---- Style -----
# python  main.py\
#  --file_in /home/anna/data_19Nov2022/old_syn\
#  --target /home/anna/data_19Nov2022/real\
#  --mode styleLinear\
#  --lr 0.00001 --device cuda:1\
#  --prefix MIA \
#  --num_epochs 5 \
#  --burnin 10 \
#  --normalization instance \
#  --batch_size 16 \
#  --weight_sides 10\
#  --weight_adv_latent 0.01 \
#  --weight_cycle 0.001 \
#  --n_iters 4 \
#  --loss l1\
#  --pretrained_style /home/anna/ResultsSignalDA/Benchmark2021-09-30_styleLinear/ \
#  --pretrained  None
### Output in: /home/anna/ResultsSignalDA//MIA2022-11-22_styleLinear/

 #  /home/anna/style_results/ReproduceNotPretrained2021-09-22_styleLinear_instance/\


# step 2 ---------- Sides ----------
# python  main.py \
#  --file_in /home/anna/data_19Nov2022/old_syn\
#  --target /home/anna/data_19Nov2022/real\
#  --mode sidesAE\
#  --lr 0.001 \
#  --device cuda:0\
#  --prefix MIA\
#  --loss l1\
#  --num_epochs 5\
#  --pretrained_style /home/anna/ResultsSignalDA//MIA2022-11-22_styleLinear/\
#  --pretrained /home/anna/style_results/adv2021-06-21_chunk-1_sides/\
#  --batch_size 16\
#  --burnin 100
### Output in: /home/anna/ResultsSignalDA//MIA2022-12-01_sidesAE/

# python  validationSignal.py --mode styleLinear --device cuda:0\
#  --file_syn /home/anna/data_19Nov2022/old_syn_test.h5\
#  --file_real /home/anna/data_19Nov2022/real_test.h5\
#  --prefix MIA \
#  --pretrained_style /home/anna/ResultsSignalDA//MIA2022-11-22_styleLinear//\
#  --pretrained_sides /home/anna/ResultsSignalDA//MIA2022-12-01_sidesAE//\
#  --tgt_dir /home/anna/ResultsSignalDA/MIAvalidation/signal/\
#  --dataset syn


# python  validationSignal.py --mode styleLinear --device cuda:0\
#  --file_syn /home/anna/data_19Nov2022/old_syn_test.h5\
#  --file_real /home/anna/data_19Nov2022/real_test.h5\
#  --prefix MIA \
#  --pretrained_style /home/anna/ResultsSignalDA//MIA2022-11-22_styleLinear//\
#  --pretrained_sides /home/anna/ResultsSignalDA//MIA2022-12-01_sidesAE//\
#  --tgt_dir /home/anna/ResultsSignalDA/MIAvalidation/signal/\
#  --dataset real
# /home/anna/ResultsSignalDA/MIAvalidation/signal//MIAstyleLinear__2022-12-03//real_54.h5

# python  validationReconstruction.py\
#  --folder /home/anna/ResultsSignalDA/MIAvalidation/signal//MIAstyleLinear__2022-12-03/\
#  --tgtfolder /home/anna/ResultsSignalDA/MIAvalidation/reconstruction//MIAstyleLinear__2022-12-03/\
#  --data syn_0\
#  --geometry multi --mode signal_with_RC --subset 0

# for i in {0..54}
# do
#    python  validationReconstruction.py\
#      --folder /home/anna/ResultsSignalDA/MIAvalidation/signal//MIAstyleLinear__2022-12-03/\
#      --tgtfolder /home/anna/ResultsSignalDA/MIAvalidation/reconstruction//MIAstyleLinear__2022-12-03/\
#      --data real\_$i\
#      --geometry multi --mode signal_with_RC --subset 0
# done

# Reconstructions of old dataset
# python  validationReconstruction.py\
#  --folder /home/anna/dlbirhoui_data/\
#  --data arm\
#  --tgtfolder /home/anna/ResultsSignalDA/MIAvalidation_olddataset/Benchmark_LinGT_EN/\
#  --geometry linear \
#  --mode sigmat_multisegment \
#  --subset 0\
#  --modes_list 'ElasticNet 1e-5'

# python  validationReconstruction.py\
#  --folder /home/anna/dlbirhoui_data/\
#  --data arm\
#  --tgtfolder /home/anna/ResultsSignalDA/MIAvalidation_olddataset/Benchmark_MultiGT_EN/\
#  --geometry multi \
#  --mode sigmat_multisegment \
#  --subset 0\
#  --modes_list 'ElasticNet 1e-5'

python  validationReconstruction.py\
 --folder /home/anna/dlbirhoui_data/\
 --data arm\
 --tgtfolder /home/anna/ResultsSignalDA/MIAvalidation_olddataset/Benchmark_MultiGT_BP/\
 --geometry multi \
 --mode sigmat_multisegment \
 --subset 0\
 --modes_list 'BackProjection'
