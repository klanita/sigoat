#!/bin/bash

# python  main.py --mode VNsides --lr 0.01 --device cuda:0\
#  --prefix ScaleSides --target_modality recon_multisegment --test True\ 
#  --pretrained_vn /home/anna/style_results/ScaleSides2021-06-28_chunk-1_VNsides/

# python  main.py --mode VNsides --lr 0.0001 --device cuda:0\
#  --prefix ScaleSides --target_modality ground_truth \
#  --pretrained_vn /home/anna/style_results/ScaleSides2021-06-29_chunk-1_VNsides/

# python  main.py --mode unet --lr 0.01 --device cuda:1 --batch_size 32\
#  --prefix Baseline --target_modality recon_multisegment

# python  main.py --mode VNsides --lr 0.01 --device cuda:1\
#  --prefix ScaleSides --target_modality ground_truth \
#  --pretrained_vn /home/anna/style_results/ScaleSides2021-06-30_recon_multisegment_VNsides/

# python  main.py --mode sides_old_pipeline --lr 0.01 --device cuda:1\
#  --prefix SidesInstance\
#  --pretrained /home/anna/style_results/SidesInstance2021-07-04_ground_truth_sides_old_pipeline/\
#  --normalization instance --batch_size 64 --test True


# python  main.py --mode sides_old_pipeline --lr 0.01 --device cuda:0\
#  --prefix NewDataset --pretrained /home/anna/style_results/NewDataset2021-07-06_ground_truth_sides_old_pipeline/\
#  --normalization batch --batch_size 32

# python  main.py --mode VNsides --lr 0.001 --device cuda:1\
#  --prefix NewDataset --target_modality recon_multisegment \
#  --pretrained /home/anna/style_results/NewDataset2021-07-07_ground_truth_sides_old_pipeline/ \
#  --pretrained_vn /home/anna/style_results/ScaleSides2021-07-02_recon_multisegment_VNsides/

# python  main.py --mode VNsides --lr 0.001 --device cuda:1\
#  --prefix NewDataset --target_modality recon_multisegment \
#  --pretrained /home/anna/style_results/NewDataset2021-07-08_ground_truth_sides_old_pipeline/ \
#  --pretrained_vn /home/anna/style_results/NewDataset2021-07-08_recon_multisegment_VNsides/

# python  main.py --mode recFinger --lr 0.001 --device cuda:1\
#  --prefix TestRec --geometry ringCup --split 0 --loss l2\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64


# python  main.py --mode recFinger --lr 0.001 --device cuda:1\
#  --prefix TestRec --geometry ringCup --split 0 --loss l1\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained /home/anna/style_results/TestRec2021-08-05_ground_truth_recFinger/\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64
# Output in: /home/anna/style_results/TestRec2021-08-06_ground_truth_recFinger/

# python  main.py --mode recFinger --lr 0.001 --device cuda:1\
#  --prefix TestRec --geometry ringCup --split 3 --loss l1\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained None\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64
#  Results saved to /home/anna/style_results/TestRec2021-08-19_ground_truth_recFinger/

# python  main.py --mode recFinger --lr 0.001 --device cuda:1\
#  --prefix TestRec --geometry ringCup --split 3 --loss l1 --test True\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained None\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64


# python  main.py --mode testFinger --lr 0.001 --device cuda:1\
#  --prefix TestRec --geometry ringCup --split 3 --loss l1 --test True\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained /home/anna/style_results/TestRec2021-08-19_ground_truth_recFinger/\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64


# python  main.py --mode recFinger --lr 0.001 --device cuda:1\
#  --prefix TestRec --geometry ringCup --split 1 --loss l1 \
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained /home/anna/style_results/TestRec2021-08-19_ground_truth_recFinger/\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64
#  /home/anna/style_results/TestRec2021-08-20_ground_truth_recFinger/

#  python  main.py --mode recFinger --lr 0.001 --device cuda:1\
#  --prefix TestRec --geometry ringCup --split 3 --loss l1 \
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained /home/anna/style_results/TestRec2021-08-19_ground_truth_recFinger/\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64

# python  main.py --mode recFinger --lr 0.001 --device cuda:1\
#  --prefix TestRec --geometry ringCup --split right --loss l1\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained None\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64
#  /home/anna/style_results/TestRec2021-08-23_ground_truth_recFinger/

#  python  main.py --mode testFinger --lr 0.001 --device cuda:1\
#  --prefix TestRec --geometry ringCup --split 3 --loss l1 --test True\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained /home/anna/style_results/TestRec2021-08-23_ground_truth_recFinger/\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64

#  python  main.py --mode recFinger --lr 0.001 --device cuda:1\
#  --prefix TestRec --geometry ringCup --split right --loss l1\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained /home/anna/style_results/TestRec2021-08-23_ground_truth_recFinger/\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64

#  python  main.py --mode styleImages --lr 0.001 --device cuda:1\
#  --prefix styleImages --loss l1 \
#  --pretrained_style None\
#  --pretrained None\
#  --normalization batch --batch_size 64


# python  main.py --mode VNsides --lr 0.0001 --device cuda:1 --batch_size 2\
#  --prefix Check --target_modality recon_multisegment \
#  --pretrained /home/anna/style_results/NewDataset2021-07-08_ground_truth_sides_old_pipeline/ \
#  --pretrained_vn /home/anna/style_results/Check2021-08-30_recon_multisegment_VNsides/
# Results saved to /home/anna/style_results/Check2021-08-27_recon_multisegment_VNsides/

# python  validation_invivo.py --mode sides_old_pipeline --lr 0.001 --device cuda:1 --batch_size 4\
#  --prefix Check --target_modality recon_multisegment --test True\
#  --pretrained /home/anna/style_results/NewDataset2021-07-08_ground_truth_sides_old_pipeline/ \
#  --pretrained_vn /home/anna/style_results/Check2021-08-30_recon_multisegment_VNsides/
#  --pretrained_vn /home/anna/style_results/Check2021-08-30_recon_multisegment_VNsides/ 
#  --pretrained_vn /home/anna/style_results/Check2021-08-29_recon_multisegment_VNsides/
#  --pretrained_vn /home/anna/style_results/Check2021-08-27_recon_multisegment_VNsides/

# python  main.py --mode sidesTwo --lr 0.001 --device cuda:1\
#  --prefix SidesTwo --split right --loss l1 \
#  --normalization batch --batch_size 64\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/

# python  validation.py --mode sidesTwo --device cuda:1 --batch_size 32\
#  --prefix NewValidation --target_modality recon_multisegment --test True\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/ \
#  --pretrained_vn /home/anna/style_results/Check2021-08-30_recon_multisegment_VNsides/


# python  main.py --mode VNsides --lr 0.00001 --device cuda:1\
#  --prefix SidesTwo --loss l1 --target_modality recon_multisegment --n_knots 35\
#  --normalization batch --batch_size 4\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/
#  --pretrained_vn /home/anna/style_results/Check2021-08-30_recon_multisegment_VNsides/

# python  main.py --mode VNsides --lr 0.001 --device cuda:1 --n_steps 5\
#  --prefix ArtifactRemoval --loss l1 --target_modality recon_ring --n_knots 35\
#  --normalization batch --batch_size 4\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/

# python  main.py --mode VNsidesLinear --lr 0.01 --device cuda:1 --n_steps 5\
#  --prefix ArtifactRemoval --loss l2 --target_modality ground_truth --n_knots 15 \
#  --normalization batch --batch_size 8\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/\
#  --pretrained_vn /home/anna/style_results/ArtifactRemoval2021-09-08_ground_truth_VNsidesLinear/


# python  validation.py --mode sidesTwo --device cuda:1 --batch_size 16 \
#  --prefix Test --target_modality recon_multisegment --test True --n_knots 35\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/\
#  --pretrained_vn /home/anna/style_results/SidesTwo2021-09-06_recon_multisegment_VNsides/

# python  validationL1.py --mode sidesTwo --device cuda:1 --batch_size 32\
#  --prefix L1 --target_modality ground_truth --test True\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/\

# python  mainStyle.py --mode styleMulti --lr 0.0001 --device cuda:1\
#  --prefix LowLR \
#  --normalization batch --batch_size 8\
#  --pretrained_style /home/anna/style_results/LowLR2021-09-13_styleMulti/

# python  mainStyle.py --mode styleLinear --lr 0.0001 --device cuda:1\
#  --prefix Ablation_WOStyleDec --num_epochs 100 --burnin 8000 \
#  --normalization instance --batch_size 16\
#  --pretrained_style None  --weight_adv 1.0 --weight_cycle 1.0
#  /home/anna/style_results/Ablation_WOStyleDec_WOgradClip2021-09-17_styleLinear_batch/

#  python  mainStyle.py --mode styleLinear --lr 0.0001 --device cuda:1\
#  --prefix Ablation_WOStyleDec --num_epochs 100 --burnin 100 \
#  --normalization instance --batch_size 16\
#  --pretrained_style None  --weight_adv 1.0 --weight_cycle 1.0
# /home/anna/style_results/Ablation_WithStyleDec_WithCycle2021-09-20_styleLinear_instance/

# python  main.py --mode VNsides --lr 0.01 --device cuda:1\
#  --prefix ScaleSides --target_modality ground_truth \
#  --pretrained_vn /home/anna/style_results/ScaleSides2021-06-30_recon_multisegment_VNsides/

#  python  mainStyle.py --mode styleLinear --lr 0.001 --device cuda:1\
#  --prefix NewSyn --num_epochs 100 --burnin 1000 \
#  --normalization instance --batch_size 16\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --pretrained_style /home/anna/style_results/deeplatent2021-06-09_chunk-1/
# /home/anna/style_results/NewSyn2021-09-20_styleLinear_instance/

# python  main.py --mode sides --lr 0.001 --device cuda:1\
#  --prefix NewSyn\
#  --pretrained_style /home/anna/style_results/NewSyn2021-09-20_styleLinear_instance/\
#  --pretrained /home/anna/style_results/adv2021-06-21_chunk-1_sides/\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --batch_size 16 
# Results saved to /home/anna/style_results/NewSyn2021-09-21_ground_truth_sides/

#  python  mainStyle.py --mode styleLinear --lr 0.001 --device cuda:1\
#  --prefix ReproduceNotPretrained --num_epochs 100 --burnin 0 \
#  --normalization instance --batch_size 16\
#  --weight_adv_latent 0.01 --n_iters 1 --loss l2\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --pretrained_style  /home/anna/style_results/ReproduceNotPretrained2021-09-21_styleLinear_instance/

#  python  mainStyle.py --mode styleLinear --lr 0.001 --device cuda:1\
#  --prefix ReproduceNotPretrained --num_epochs 100 --burnin 0 \
#  --normalization instance --batch_size 16\
#  --weight_adv_latent 0.01 --n_iters 1 --loss l2\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --pretrained_style  /home/anna/style_results/ReproduceNotPretrained2021-09-22_styleLinear_instance/

# python  main.py --mode sides --lr 0.0001 --device cuda:1\
#  --prefix NewSynPretrainedEnc --loss l1\
#  --pretrained_style /home/anna/style_results/ReproduceNotPretrained2021-09-22_styleLinear_instance/\
#  --pretrained None\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --batch_size 16 --burnin 1000


#  python  mainStyle.py --mode styleLinear --lr 0.001 --device cuda:1\
#  --prefix Finger --dataset Finger --num_epochs 100 --burnin 0 \
#  --normalization instance --batch_size 16\
#  --weight_adv_latent 0.01 --n_iters 1 --loss l2\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --pretrained_style  /home/anna/style_results/ReproduceNotPretrained2021-09-22_styleLinear_instance/


#  python  mainStyle.py --mode styleLinear --lr 0.001 --device cuda:1\
#  --prefix Finger --dataset Finger --num_epochs 100 --burnin 0 \
#  --normalization instance --batch_size 16\
#  --weight_adv_latent 0.01 --n_iters 1 --loss l1\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --pretrained_style  /home/anna/style_results/Finger2021-09-23_styleLinear_instance/
# Results saved to /home/anna/style_results/Finger2021-09-24_styleLinear_instance/

#  python  main.py --mode styleLinear --lr 0.001 --device cuda:1\
#  --prefix Test --dataset Finger --num_epochs 20 --burnin 50 --test True\
#  --normalization instance --batch_size 16\
#  --weight_adv_latent 0.001 --n_iters 1 --loss l1 --weight_adv 0.01\
#  --pretrained_style  /home/anna/style_results/deeplatent2021-06-09_chunk-1/
# #  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
# /home/anna/style_results/FingerOldSyn2021-09-27_styleLinear_instance/                                                                                                           

#  python  main.py --mode styleFull --lr 0.001 --device cuda:1\
#  --prefix FullModel --num_epochs 100 --burnin 0 --test True\
#  --normalization batch --batch_size 8 \
#  --weight_adv_latent 0.001 --n_iters 1 --loss l1\
#  --pretrained_style  None --pretrained  /home/anna/style_results/FullModel2021-09-26_styleFull_batch/

# python  main.py --mode sidesAE --lr 0.001 --device cuda:1\
#  --prefix Finger --dataset Finger --loss l1 \
#  --pretrained_style /home/anna/style_results/FingerOldSyn2021-09-27_styleLinear_instance/\
#  --pretrained /home/anna/style_results/adv2021-06-21_chunk-1_sides/\
#  --batch_size 16 --burnin 100
# /home/anna/style_results//Finger2021-09-27_sidesAE/

python  main.py --mode sidesAE --lr 0.001 --device cuda:1 --test True\
 --prefix BS32_ --dataset Finger --loss l1 \
 --pretrained_style /home/anna/style_results/FingerOldSyn2021-09-27_styleLinear_instance/\
 --pretrained None\
 --batch_size 32 --burnin 1000
