#!/bin/bash

# python  validationReconstruction.py\
#  --folder /home/anna/dlbirhoui_data/\
#  --data arm\
#  --tgtfolder /home/anna/ResultsSignalDA/MIAvalidation_olddataset/Benchmark_LinearGT_EN/\
#  --geometry linear \
#  --mode sigmat_multisegment \
#  --subset 0\
#  --modes_list 'ElasticNet 1e-5'


 python  validationReconstruction.py\
 --folder /home/anna/dlbirhoui_data/\
 --data arm\
 --tgtfolder /home/anna/ResultsSignalDA/MIAvalidation_olddataset/Benchmark_LinearGT_BP/\
 --geometry linear \
 --mode sigmat_multisegment \
 --subset 0\
 --modes_list 'BackProjection'
