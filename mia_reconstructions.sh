#!/bin/bash
# for i in {0..11}
# do
# start=$(((i-1)*500))
# finish=$((i*500))
# nohup python  validationReconstruction.py\
#     --folder /home/anna/data_19Nov2022/\
#     --tgtfolder MIAreview/Syn/Train/\
#     --data old_syn_train\
#     --geometry linear \
#     --mode sigmat_multisegment\
#     --modes_list 'ElasticNet 1e-5'\
#     --subset 0\
#     --img_start $start\
#     --img_finish $finish >> MIAreview/fastrec_linear.txt &
# done



# nohup python  validationReconstruction.py\
#     --folder /home/anna/data_19Nov2022/\
#     --tgtfolder MIAreview/Syn/Val/\
#     --data old_syn_val\
#     --geometry linear \
#     --mode sigmat_multisegment\
#     --modes_list 'ElasticNet 1e-5'\
#     --subset 0\
#     --img_start 0\
#     --img_finish -1 >> MIAreview/fastrec_linear.txt &

nohup python  validationReconstruction.py\
    --folder /home/anna/data_19Nov2022/\
    --tgtfolder MIAreview/Syn/Test/\
    --data old_syn_test\
    --geometry linear \
    --mode sigmat_multisegment\
    --modes_list 'ElasticNet 1e-5'\
    --subset 0\
    --img_start 0\
    --img_finish -1 >> MIAreview/fastrec_linear.txt &

nohup python  validationReconstruction.py\
    --folder /home/anna/data_19Nov2022/\
    --tgtfolder MIAreview/Syn/Val/\
    --data old_syn_val\
    --geometry multi \
    --mode sigmat_multisegment\
    --modes_list 'ElasticNet 1e-5'\
    --subset 0\
    --img_start 0\
    --img_finish -1 >> MIAreview/fastrec_linear.txt &


nohup python  validationReconstruction.py\
    --folder /home/anna/data_19Nov2022/\
    --tgtfolder MIAreview/Syn/Test/\
    --data old_syn_test\
    --geometry multi \
    --mode sigmat_multisegment\
    --modes_list 'ElasticNet 1e-5'\
    --subset 0\
    --img_start 0\
    --img_finish -1 >> MIAreview/fastrec_linear.txt &