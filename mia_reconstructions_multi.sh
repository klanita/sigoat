#!/bin/bash

# python  validationReconstruction.py\
#     --folder /home/anna/data_19Nov2022/\
#     --tgtfolder MIAreview/Syn/Train/\
#     --data old_syn_train\
#     --geometry linear \
#     --mode sigmat_multisegment\
#     --modes_list 'ElasticNet 1e-5'\
#     --subset 0


# for i in {0..11}
# do
# start=$(((i-1)*500))
# finish=$((i*500))
# nohup python  validationReconstruction.py\
#     --folder /home/anna/data_19Nov2022/\
#     --tgtfolder MIAreview/Syn/Train/\
#     --data old_syn_train\
#     --geometry multi \
#     --mode sigmat_multisegment\
#     --modes_list 'ElasticNet 1e-5'\
#     --subset 0\
#     --img_start 0\
#     --img_start $start\
#     --img_finish $finish >> MIAreview/fastrec_multi.txt &
# done

i=11
start=$(((i-1)*500))
finish=$((i*500))
nohup python  validationReconstruction.py\
    --folder /home/anna/data_19Nov2022/\
    --tgtfolder MIAreview/Syn/Train/\
    --data old_syn_train\
    --geometry multi \
    --mode sigmat_multisegment\
    --modes_list 'ElasticNet 1e-5'\
    --subset 0\
    --img_start $start\
    --img_finish $finish >> MIAreview/fastrec_multi.txt &