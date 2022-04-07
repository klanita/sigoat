#!/bin/bash

# main network
# python  validationSignal.py --mode sides_old_pipeline --device cpu\
#  --prefix NewLinearInput_ --tgt_dir ../Review 

# python  validationReconstruction.py\
#  --folder TBD\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 1024 --subset 0

# python  validationReconstruction.py\
#  --folder TBD\
#  --geometry multi --data Real --mode signal_with_denoise --nimgs 1024 --subset 0

# # benchmark sides
# python  validationSignal.py --mode sidesTwo --device cpu\
#  --prefix NewSides_ --tgt_dir "/home/berkan/docs/phd/midl/code/git/OptoAcoustics/data" \
#  --pretrained_sides /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/


# python  validationReconstruction.py\
#  --folder TBD\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 1024 --subset 0


# # two more bm from Anna
# python  validationReconstruction.py\
#  --folder ./validation//Behnch1024_woStyle_L2_full_styleNone_2022-02-10/\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 1024 --subset 0


# python  validationReconstruction.py\
#  --folder ./validation//Behnch1024_woStyleSyn_L2_full_styleNone_2022-02-10/\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 1024 --subset 0



# Behnch1024_woStyle_L2_full_styleNone_2022-02-10/
# Behnch1024_woStyleSyn_L2_full_styleNone_2022-02-10/
# NewSides_sidesTwo__2022-02-10/

# NewLinearInput_sides_old_pipeline__2022-02-09/

# python  validationReconstruction.py\
#  --folder /home/anna/dlbirhoui_data/\
#  --data arm\
#  --mode sigmat_multisegment\
#  --geometry multi --nimgs 1024 --subset 0

python  validationReconstruction.py\
 --folder /home/anna/dlbirhoui_data/\
 --tgtfolder ./MIDL/GT/linear/\
 --data arm\
 --mode sigmat_multisegment\
 --geometry linear --nimgs 1024 --subset 0