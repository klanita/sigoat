#!/bin/bash

# python  main.py --mode VNsides --lr 0.01 --device cuda:0\
#  --prefix ScaleSides --target_modality recon_multisegment --test True\ 
#  --pretrained_vn /home/anna/style_results/ScaleSides2021-06-28_chunk-1_VNsides/

# python  main.py --mode VNsides --lr 0.0001 --device cuda:0\
#  --prefix ScaleSides --target_modality ground_truth \
#  --pretrained_vn /home/anna/style_results/ScaleSides2021-06-29_chunk-1_VNsides/

# python  main.py --mode unet --lr 0.01 --device cuda:0 --batch_size 64\
#  --prefix Baseline --target_modality ground_truth \

# python  main.py --mode VNsides --lr 0.01 --device cuda:0\
#  --prefix ScaleSides --target_modality recon_multisegment \
#  --pretrained_vn /home/anna/style_results/ScaleSides2021-06-30_recon_multisegment_VNsides/


# python  main.py --mode sides_old_pipeline --lr 0.01 --device cuda:0\
#  --prefix SidesBatch \
#  --pretrained None --normalization batch --batch_size 64

# python  main.py --mode sides_old_pipeline --lr 0.001 --device cuda:0\
#  --prefix NewDataset --pretrained /home/anna/style_results/NewDataset2021-07-06_ground_truth_sides_old_pipeline/\
#  --normalization batch --batch_size 32


# python  main.py --mode style --lr 0.001 --device cuda:0\
#  --prefix TestStyle --geometry ringCup.mat \
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 16

# python  main.py --mode style --lr 0.001 --device cuda:0\
#  --prefix TestStyle --geometry ringCup\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-07-30_ground_truth_style/\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 16 --split 0

# python  main.py --mode recFinger --lr 0.001 --device cuda:0\
#  --prefix TestRec --geometry ringCup --split 1 --loss l2\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64
#  Results saved to /home/anna/style_results/TestRec2021-08-05_ground_truth_recFinger/

# python  main.py --mode recFinger --lr 0.001 --device cuda:0\
#  --prefix TestRec --geometry ringCup --split 2 --loss l2\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained None\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64
#  Results saved to /home/anna/style_results/TestRec2021-08-19_ground_truth_recFinger/

#  python  main.py --mode recFinger --lr 0.001 --device cuda:0\
#  --prefix TestRec --geometry ringCup --split 0 --loss l1 \
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained /home/anna/style_results/TestRec2021-08-19_ground_truth_recFinger/\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64
#  /home/anna/style_results/TestRec2021-08-22_ground_truth_recFinger/

# python  main.py --mode recFinger --lr 0.001 --device cuda:0\
#  --prefix TestRec --geometry ringCup --split left --loss l1\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained None\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64
 
#  python  main.py --mode recFinger --lr 0.001 --device cuda:0\
#  --prefix TestRec --geometry ringCup --split left --loss l1\
#  --file_in /home/anna/data/parsed_simulated_ellipsesSkinMask_mgt_ms_ring_256_ratio_09_1_20210412_anna.h5\
#  --pretrained_style /home/anna/style_results/TestStyle2021-08-04_ground_truth_style/\
#  --pretrained /home/anna/style_results/TestRec2021-08-23_ground_truth_recFinger/\
#  --target /home/firat/docs/dlbirhoui/parsed_data/parsed_fingerBP_and_signals.h5\
#  --normalization batch --batch_size 64

# python  main.py --mode VNsides --lr 0.0001 --device cuda:0 --batch_size 2\
#  --prefix Check --target_modality ground_truth \
#  --pretrained /home/anna/style_results/NewDataset2021-07-08_ground_truth_sides_old_pipeline/ \
# /home/anna/style_results/Check2021-08-27_ground_truth_VNsides/

# python  validation_invivo.py --mode VNsides --lr 0.001 --device cuda:0 --batch_size 4\
#  --prefix Check --target_modality recon_multisegment --test True\
#  --pretrained /home/anna/style_results/NewDataset2021-07-08_ground_truth_sides_old_pipeline/ \
#  --pretrained_vn /home/anna/style_results/Check2021-08-27_ground_truth_VNsides/

# python  validation_invivo.py --mode VNsides --lr 0.001 --device cuda:0 --batch_size 4\
#  --prefix Check --target_modality ground_truth --test True\
#  --pretrained /home/anna/style_results/NewDataset2021-07-08_ground_truth_sides_old_pipeline/ \
#  --pretrained_vn /home/anna/style_results/Check2021-08-27_ground_truth_VNsides/

# python  main.py --mode sidesTwo --lr 0.001 --device cuda:0\
#  --prefix lr0001 --split left --loss l1 \
#  --normalization batch --batch_size 16\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/\
#  --pretrained_vn /home/anna/style_results/SidesTwo2021-09-01_recon_multisegment_VNsides/

# python  validation.py --mode sidesTwo --device cuda:0 --batch_size 8\
#  --prefix lr0001 --target_modality recon_multisegment --test True\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/\
#  --pretrained_vn /home/anna/style_results/SidesTwo2021-09-01_recon_multisegment_VNsides/


#  python  main.py --mode VNsides --lr 0.0001 --device cuda:0 --n_steps 4\
#  --prefix ArtifactRemoval --loss l1 --target_modality ground_truth --n_knots 35\
#  --normalization batch --batch_size 4\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/

#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/

# python  validation.py --mode sidesTwo --device cuda:0 --batch_size 32\
#  --prefix lr001 --target_modality recon_multisegment --test True\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/\
#  --pretrained_vn /home/anna/style_results/SidesTwo2021-09-02_recon_multisegment_VNsides/

# python  validation.py --mode sidesTwo --device cuda:0 --batch_size 32 --n_steps 2\
#  --prefix Scaler --target_modality recon_multisegment --test True --n_knots 35\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/\
#  --pretrained_vn /home/anna/style_results/SidesTwo2021-09-06_recon_multisegment_VNsides_2stpes/


# python  validation.py --mode sidesTwo --device cuda:0 --batch_size 8 --n_steps 4\
#  --prefix Scaler --target_modality recon_multisegment --test True --n_knots 35\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/\
#  --pretrained_vn /home/anna/style_results/ArtifactRemoval2021-09-07_ground_truth_VNsides/

# python  validation.py --mode sidesTwo --device cuda:0 --batch_size 32 --n_steps 4\
#  --prefix L1 --target_modality ground_truth --test True --n_knots 35\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/\
#  --pretrained_vn /home/anna/style_results/ArtifactRemoval2021-09-07_ground_truth_VNsides/

# python  validation.py --mode sidesTwo --device cuda:0 --batch_size 8 --n_steps 5\
#  --prefix ArtifactRemoval --target_modality ground_truth --test True --n_knots 35\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/\
#  --pretrained_vn /home/anna/style_results/ArtifactRemoval2021-09-07_recon_ring_VNsides/

# python  main.py --mode VNsidesLinear --lr 0.001 --device cuda:0 --n_steps 10\
#  --prefix ArtifactRemoval --loss l2 --target_modality ground_truth --n_knots 35 \
#  --normalization batch --batch_size 8\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/


# python  validationLinear.py --mode sidesTwo --device cuda:0 --batch_size 8 --n_steps 10 \
#  --prefix Test --target_modality recon_multisegment --test True --n_knots 30\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/\
#  --pretrained_vn /home/anna/style_results/ArtifactRemoval2021-09-08_ground_truth_VNsidesLinear/

# python  main.py --mode VNsidesLinear --lr 0.001 --device cuda:0 --n_steps 10\
#  --prefix WihoutTV --loss l2 --target_modality ground_truth --n_knots 35 \
#  --normalization batch --batch_size 8\
#  --pretrained /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/

# python  mainStyle.py --mode styleMulti --lr 0.001 --device cuda:0\
#  --prefix HighLR \
#  --normalization batch --batch_size 8\
#  --pretrained_style /home/anna/style_results/HighLR2021-09-13_styleMulti/


# python  mainStyle.py --mode styleLinear --lr 0.001 --device cuda:0\
#  --prefix Ablation_WOStyleDec --num_epochs 100 --burnin 1000 \
#  --normalization batch --batch_size 16\
#  --pretrained_style None\
#  --weight_adv 0.0001 --weight_cycle 1.0
#  /home/anna/style_results/Ablation_WOStyleDec2021-09-16_styleLinear_batch/
#  --pretrained_style /home/anna/style_results/HighLR2021-09-13_styleMulti/
# /home/anna/style_results/Ablation_WOStyleDec2021-09-16_styleLinear_batch/

# python  mainStyle.py --mode styleLinear --lr 0.001 --device cuda:0\
#  --prefix Ablation_WithStyleDec_WOgradClip --num_epochs 100 --burnin 1000 \
#  --normalization instance --batch_size 16\
#  --pretrained_style None
#  /home/anna/style_results/Ablation_WithStyleDec_WOgradClip2021-09-18_styleLinear_instance/

#  python  mainStyle.py --mode styleLinear --lr 0.001 --device cuda:0\
#  --prefix Ablation_WithStyleDec_WithCycle --num_epochs 100 --burnin -1 \
#  --normalization instance --batch_size 16\
#  --pretrained_style /home/anna/style_results/Ablation_WithStyleDec_WOgradClip2021-09-18_styleLinear_instance/ 

#  python  mainStyle.py --mode styleLinear --lr 0.001 --device cuda:0\
#  --prefix Reproduce --num_epochs 100 --burnin 1000 \
#  --normalization instance --batch_size 16\
#  --pretrained_style  /home/anna/style_results/Reproduce2021-09-20_styleLinear_instance/
#  Results saved to /home/anna/style_results/Reproduce2021-09-20_styleLinear_instance/

#  python  mainStyle.py --mode styleLinear --lr 0.001 --device cuda:0\
#  --prefix Reproduce --num_epochs 100 --burnin 0 \
#  --normalization instance --batch_size 16\
#  --weight_adv_latent 0.0001 --n_iters 2\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --pretrained_style  /home/anna/style_results/Reproduce2021-09-20_styleLinear_instance/
# Results saved to /home/anna/style_results/Reproduce2021-09-21_styleLinear_instance/

# python  main.py --mode sides --lr 0.001 --device cuda:0\
#  --prefix NewSyn\
#  --pretrained_style /home/anna/style_results/NewSyn2021-09-20_styleLinear_instance/\
#  --pretrained /home/anna/style_results/adv2021-06-21_chunk-1_sides/\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --batch_size 16 --test True

# python  main.py --mode sides --lr 0.001 --device cuda:0\
#  --prefix NewSynFromScratch\
#  --pretrained_style /home/anna/style_results/NewSyn2021-09-20_styleLinear_instance/\
#  --pretrained None\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --batch_size 8 --burnin -1

# python  main.py --mode sides --lr 0.001 --device cuda:0\
#  --prefix NewSynFromScratch\
#  --pretrained_style /home/anna/style_results/NewSyn2021-09-20_styleLinear_instance/\
#  --pretrained None\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --batch_size 8 --burnin -1

# python  main.py --mode sides --lr 0.0001 --device cuda:0\
#  --prefix NewSynFromScratch --loss l1\
#  --pretrained_style /home/anna/style_results/ReproduceNotPretrained2021-09-22_styleLinear_instance/\
#  --pretrained None\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --batch_size 8 --burnin 1000


# python  main.py --mode sides --lr 0.001 --device cuda:0\
#  --prefix NewSynFromScratch --loss l1\
#  --pretrained_style /home/anna/style_results/ReproduceNotPretrained2021-09-22_styleLinear_instance/\
#  --pretrained None\
#  --file_in /home/firat/docs/dlbirhoui/parsed_data/parsed_simulated_cylindersSkinMask_mgt_ms_ring_256_20210507.h5\
#  --batch_size 16 --burnin 500
#  Results saved to /home/anna/style_results/NewSynFromScratch2021-09-23_ground_truth_sides/

#  python  mainStyle.py --mode styleFull --lr 0.001 --device cuda:0\
#  --prefix FullModel --num_epochs 100 --burnin 500 \
#  --normalization batch --batch_size 8 \
#  --weight_adv_latent 0.001 --n_iters 1 --loss l2\
#  --pretrained_style  None --pretrained  None

#  python  main.py --mode styleFull --lr 0.001 --device cuda:0\
#  --prefix FullModelL1 --num_epochs 100 --burnin 10 \
#  --normalization batch --batch_size 16 --weight_sides 100\
#  --weight_adv_latent 0.001 --weight_cycle 0.0001 --weight_mmd 0.0001\
#  --n_iters 4 --loss l1\
#  --pretrained_style  None\
#  --pretrained  /home/anna/style_results/FullModel2021-09-27_styleFull_batch/
# Results saved to /home/anna/ResultsSignalDA//FullModelL12021-09-28_styleFull/

 python  main.py --mode styleFull --lr 0.001 --device cuda:1\
 --prefix FullModelL2 --num_epochs 100 --burnin 50 \
 --normalization batch --batch_size 16 --weight_sides 10\
 --weight_adv_latent 0.001 --weight_cycle 0.001 \
 --n_iters 4 --loss l2\
 --pretrained_style  None\
 --pretrained  /home/anna/ResultsSignalDA//FullModelL12021-09-28_styleFull/
