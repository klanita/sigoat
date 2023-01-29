#!/bin/bash

# for i in {11..20}
# do
# echo $i
# # nohup echo $i >> fastrec.txt &
# nohup python  validationReconstruction.py\
#     --folder /home/anna/ResultsSignalDA/MIAvalidation_olddataset/signal//MIAstyleLinear__2022-12-09/\
#     --tgtfolder /home/anna/ResultsSignalDA/MIAvalidation_olddataset/reconstruction//MIAstyleLinear__2022-12-09/\
#     --data real\_$i\
#     --geometry multi --mode signal_with_RC --subset 0 >> fastrec_2.txt &
# done


# for i in {0..21}
# do
# echo $i
# # nohup echo $i >> fastrec.txt &
# nohup python  validationReconstruction.py\
#     --folder /home/anna/ResultsSignalDA/MIAvalidation_olddataset/signal//MIAstyleLinear__2022-12-09/\
#     --tgtfolder /home/anna/ResultsSignalDA/MIAvalidation_olddataset/reconstruction//MIAstyleLinear__2022-12-09/\
#     --data real\_$i\
#     --geometry linear --mode signal_with_denoise --subset 0 >> fastrec.txt &
# done

for i in {1..21}
do
# nohup echo $i >> fastrec.txt &
nohup python  validationReconstruction.py\
    --folder /home/anna/ResultsSignalDA/MIAvalidation_olddataset/signal//MIAstyleLinear__2022-12-15/\
    --tgtfolder /home/anna/ResultsSignalDA/MIAvalidation_olddataset/reconstruction//MIAstyleLinear__2022-12-15/\
    --data real\_$i\
    --geometry multi --mode signal_with_RC --subset 0 >> fastrec.txt &
done