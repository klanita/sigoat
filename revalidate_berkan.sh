#!/bin/bash

# # main network
# python  validationSignal.py --mode sides_old_pipeline --device cpu\
#  --prefix NewLinearInput_ --tgt_dir "./validation/" 

# python  validationReconstruction.py\
#  --folder "./validation/NewLinearInput_sides_old_pipeline__2022-02-09" \
#  --geometry multi --data Real --mode signal_with_RC --nimgs 1024 --subset 0

python  validationReconstruction.py\
 --folder "./validation/NewLinearInput_sides_old_pipeline__2022-02-09" \
 --geometry multi --data Real --mode signal_with_denoise --nimgs 1024 --subset 0

# # benchmark sides
# python  validationSignal.py --mode sidesTwo --device cpu\
#  --prefix NewSides_ --tgt_dir "/home/berkan/docs/phd/midl/code/git/OptoAcoustics/data" \
#  --pretrained_sides /home/anna/style_results/SidesTwo2021-08-31_ground_truth_sidesTwo/


# python  validationReconstruction.py\
#  --folder TBD\
#  --geometry multi --data Real --mode signal_with_RC --nimgs 1024 --subset 0


# two more bm from Anna
