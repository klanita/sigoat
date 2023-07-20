#!/bin/bash

# for n in BN
# do
# for l in l1 l2
# do
# echo $l $n
# python ./main.py --loss_name $l 
# done
# done

# python ./main.py --loss_name l1 \
#  --file_in '/home/anna/ResultsSignalDA/GT_syn/old_syn'\
#  --test_prefix syn\
#  --mode test\
#  --ckpt /home/anna/ResultsSignalDA/UNet-BM/maxpool_l1_BN_2022-12-12/epoch=4-step=6075.ckpt

# python ./main.py --loss_name l1 \
#  --file_in '/home/anna/ResultsSignalDA/GT_real/arm'\
#  --test_prefix arm\
#  --mode test\
#  --batch_size 32\
#  --ckpt /home/anna/ResultsSignalDA/UNet-BM/maxpool_l1_BN_2022-12-12/epoch=4-step=6075.ckpt


# python ./main.py --loss_name l1 \
#  --file_in '/home/anna/ResultsSignalDA/GT_syn/old_syn'\
#  --test_prefix syn\
#  --prefix Tune\
#  --mode train\
#  --ckpt /home/anna/ResultsSignalDA/UNet-BM/maxpool_l1_BN_2022-12-12/epoch=4-step=6075.ckpt


# python ./main.py --loss_name l1 \
#  --prefix CorrectScale\
#  --test_prefix syn\
#  --mode train

#  python ./main.py --loss_name l1 \
#  --file_in '/home/anna/ResultsSignalDA/GT_real/arm'\
#  --test_prefix arm\
#  --prefix CorrectScale\
#  --mode test\
#  --batch_size 32\
#  --ckpt /home/anna/ResultsSignalDA/UNet-BM/CorrectScale_l1_BN_2022-12-12/epoch=10-step=14175.ckpt

 
#  python ./main.py --loss_name l1 \
#  --prefix NotBilinear\
#  --bilinear 0\
#  --test_prefix syn\
#  --mode train

#  python ./main.py --loss_name l1 \
#  --file_in '/home/anna/ResultsSignalDA/GT_real/arm'\
#  --test_prefix arm\
#  --prefix NotBilinear\
#  --bilinear 0\
#  --mode test\
#  --batch_size 32\
#  --ckpt /home/anna/ResultsSignalDA/UNet-BM/NotBilinear_l1_BN_2022-12-12/epoch=13-step=18900.ckpt


#  nohup 
#  python ./main.py --loss_name l1 \
#  --prefix NotBilinear\
#  --bilinear 0\
#  --test_prefix syn\
#  --mode train 
#  >> results.txt &



 python ./main.py --loss_name l1 \
 --prefix NotBilinear\
 --bilinear 0\
 --test_prefix syn\
 --mode test \
 --ckpt ../MIAreview/Results/NotBilinear_l1_BN_2023-07-07/epoch=14-step=20250.ckpt
