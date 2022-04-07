#!/bin/bash

# python  validationSignal.py --mode sides_old_pipeline --device cuda:1\
#  --prefix LinearInput_

# python  validationReconstruction.py\
#  --folder /home/anna/OptoAcoustics/validation/LinearInput_sides_old_pipeline__2021-09-20\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 1 --subset 0

# python  validationSignal.py --mode sides_old_pipeline --device cuda:1\
#  --prefix LinearInput_ 

# python  validationSignal.py --mode sides_old_pipeline --device cuda:1\
#  --prefix LinearInput_ --pretrained_style /home/anna/style_results/Ablation_WithStyleDec_WithCycle2021-09-20_styleLinear_instance/

# python  validationSignal.py --mode sidesTwo --device cuda:1\
#  --prefix LinearInput_\
#  --pretrained_sides /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/


# python  validationSignal.py --mode multi --device cuda:1\
#  --prefix MultiInput_\
#  --pretrained_style /home/anna/style_results/HighLR2021-09-13_styleMulti/

# python  validationSignal.py --mode sides_old_pipeline --device cuda:0\
#  --prefix NewSyn_\
#  --pretrained_style /home/anna/style_results/NewSyn2021-09-20_styleLinear_instance/\
#  --pretrained_sides /home/anna/style_results/adv2021-06-21_chunk-1_sides/
#  --pretrained_sides /home/anna/style_results/NewSyn2021-09-21_ground_truth_sides/

# python  validationReconstruction.py\
#  --folder /home/anna/OptoAcoustics/validation/LinearInput_sides_old_pipeline__2021-09-20\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 1 --subset 0


# python  validationSignal.py --mode full --device cuda:0\
#  --prefix Full_\
#  --pretrained_style /home/anna/style_results/FullModel2021-09-26_styleFull_batch/

# python  validationReconstruction.py\
#  --folder /home/anna/OptoAcoustics/validation/Full_full__2021-09-27/\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 32 --subset 1

# python  validationSignal.py --mode full --device cuda:0\
#  --prefix L2L1_\
#  --pretrained_style /home/anna/style_results/FullModel2021-09-27_styleFull_batch/

# python  validationReconstruction.py\
#  --folder /home/anna/OptoAcoustics/validation/L2L1_full__2021-09-27/\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 32 --subset 1



#===================== MultiSegment Style =====================
# python  validationSignal.py --mode styleMulti --device cuda:0\
#  --prefix Benchmark \
#  --pretrained_style /home/anna/ResultsSignalDA//Benchmark2021-09-30_styleMulti/

# python  validationReconstruction.py\
#  --folder /home/anna/OptoAcoustics/validation//BenchmarkstyleMulti__2021-09-30/\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 32 --subset 0

#===================== Linear Style =====================
# python  validationSignal.py --mode styleLinear --device cuda:0\
#  --prefix Benchmark \
#  --pretrained_style /home/anna/style_results/deeplatent2021-06-09_chunk-1/\
#  --pretrained_sides /home/anna/ResultsSignalDA//Benchmark_2021-09-30_sidesAE/

# python  validationReconstruction.py\
#  --folder ./validation//BenchmarkstyleLinear__2021-09-30/\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 32 --subset 0

# python  validationReconstruction.py\
#  --folder ./validation//BenchmarkstyleLinear__2021-09-30/ --scale_val 0.0\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 32 --subset 0


# python  validationReconstruction.py\
#  --folder ./validation//BenchmarkstyleLinear__2021-09-30/ --scale_val 0.0\
#  --geometry multi --data Real --mode signal_with_denoise --nimgs 32 --subset 0

#  python  validationReconstruction.py\
#  --folder /home/anna/OptoAcoustics/validation//BenchmarkstyleLinear__2021-09-30/\
#  --geometry linear --data Real --mode signal_with_denoise --nimgs 32 --subset 0


#===================== Full Style =====================
# python  validationSignal.py --mode styleFull --device cuda:0\
#  --prefix Benchmark --epoch 40\
#  --pretrained_style /home/anna/ResultsSignalDA//FullModelL22021-09-30_styleFull/

# python  validationReconstruction.py\
#  --folder /home/anna/OptoAcoustics/validation///BenchmarkstyleFull40__2021-09-30/\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 32 --subset 1

#===================== Simple sides benchmark =====================

# python  validationReconstruction.py\
#  --folder /home/anna/OptoAcoustics/validation///LinearInput_sidesTwo__2021-09-20/\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 32 --subset 0



# python  pl_validationSignal.py --mode full --device cuda:0\
#  --prefix Behnch1024_woStyle_L2_\
#  --pretrained_sides /home/anna.susmelj@ad.biognosys.ch/MIDL/results/woStyleL2Real2022-02-07_sidesAE/epoch=51-step=883.ckpt\
#  --tgt_dir ./validation/\
#  --path /home/anna.susmelj@ad.biognosys.ch/MIDL/
# ./validation//Behnch_woStyle_L2_full_styleNone_2022-02-08/

# python  pl_validationSignal.py --mode full --device cuda:0\
#  --prefix Behnch1024_woStyle_L2_\
#  --pretrained_sides /home/anna.susmelj@ad.biognosys.ch/MIDL/results/woStyleL2Real2022-02-09_sidesAE/epoch=59-step=719.ckpt\
#  --tgt_dir ./validation/\
#  --path /home/anna.susmelj@ad.biognosys.ch/MIDL/

python  pl_validationSignal.py --mode full --device cuda:0\
 --prefix Behnch1024_woStyleSyn_L2_\
 --pretrained_sides /home/anna.susmelj@ad.biognosys.ch/MIDL/results/woStyleL2L1OnlySyn2022-02-08_sidesAE/epoch=15-step=239.ckpt\
 --tgt_dir ./validation/\
 --path /home/anna.susmelj@ad.biognosys.ch/MIDL/
#  ./validation//Behnch_woStyleSyn_L2_full_styleNone_2022-02-08/

# python  validationReconstruction.py\
#  --folder ./validation//Behnch_woStyle_L2_full_styleNone_2022-02-07/\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 32 --subset 0


# python  validationReconstruction.py\
#  --folder ./validation//Behnch_woStyleSyn_L2_full_styleNone_2022-02-08/\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 32 --subset 0
