# Signal Domain Learning in Optoacoustics

This is the GitHub repository for MIDL 2022 paper "[Signal Domain Learning Approach for Optoacoustic Image Reconstruction from Limited View Data](https://openreview.net/forum?id=9NOyrfUBtx1)" that is accepted for oral presentation.

The paper proposes a "Style Network" to reduce domain gap between simulated and experimental optoacoustic data. After the reduction of domain gap, another network called "Side Network" is trained on the simulated data to remove limited view artifacts in optoacoustic imaging. Then, the trained "Side Network" is applied on the experimental data.

Conference webpage: [MIDL 2022](https://2022.midl.io/)  
All accepted papers: [OpenReview MIDL](https://openreview.net/group?id=MIDL.io/2022/Conference)

## Package Structure

### Main scripts
* [`main`](main.py) contains scripts to train the model. This is a generic call for different options.
* [`trainFull`](trainFull.py) full end-to-end training with DA and sides prediction at the same time.
* [`trainStyle`](trainStyle.py) train style on linear or multisegment parts.
* [`trainSidesDA`](trainSidesDA.py) AE for the prediction of sides.

### Supplementary scripts
* [`options`](options.py) all options for training.
* [`utils`](utils.py) various related functions. Should be cleaned a bit.
* [`model`](model.py) all versions of the networks.
* [`dataLoader`](dataLoader.py) generic data loader for different types of data (signal vs image). It crops and scales the images inside.

### Benchmarks
* [`trainStyleImage`](trainStyleImage.py) train same procedure but on images instead of signal.
* [`trainSidesRegression`](trainSidesRegression.py) train predictions of the sides without DA.

### Validation scripts
* [`validationModel`](validationModel.py) Validation of the signal. Produces .h5 file with predictions on test set.
* [`validationReconstruction`](validationReconstruction.py) Validation of the reconstruction from the signal.


### Model Based Reconstruction scripts (no training)
* [`reconstructions`](reconstructions.py) TBD.
* [`ReconstructionBP`](ReconstructionBP.py) BackProjection scripts from Berkan.
* [`ReconstructionMB`](ReconstructionMB.py) Linear model based reconstruction scripts from Berkan.


### Main Notebooks
* [`PaperFigure`](notebooks/PaperFigure.ipynb) Main plots from the paper.
* [`Show signal`](notebooks/ShowSignal.ipynb) Look at the example of reconstructed signal.
* [`Benchmark from Firat`](notebooks/BenchmarkFromFirat.ipynb) Convert results from Firat to my plotting format.

### Bash files
* [`pipeline0`](pipeline0.sh) History of calls.
* [`pipeline1`](pipeline1.sh) History of calls.
* [`validation.sh`](validation.sh) Call for scripts to perform generic validation of the model.
* [`benchmarks.sh`](benchmarks.sh) Benchmarks necessary for the ablation study.


## Parameters
### --mode
* [`--mode styleFull`](trainFull.py) End-to-end DA and predictions.
* [`--mode styleLinear`](trainStyle.py) Style transfer/DA only on linear parts.
* [`--mode styleMulti`](trainStyle.py) Style transfer/DA only on multisegment parts.
* [`--mode styleImages`](trainStyleImage.sh) Everything in image domain.
* [`--mode sidesAE`](trainSidesDA.py) AE for sides predictions.
* [`--mode sidesTwo`](trainSidesRegression.py) Simple predictions.

### --dataset
* [`--dataset Forearm`](options.py) All path to the forearm data.
* [`--dataset Finger`](options.py) All path to the finger data.

### Optional Parameters
You can find a full list of parameters with defaults and comments in: [`options`](options.py).

## Usage

 `python  main.py --mode styleFull --lr 0.001 --device cuda:0\
 --prefix FullModelL1 --num_epochs 100 --burnin 0 \
 --normalization batch --batch_size 16 --weight_sides 10\
 --weight_adv_latent 0.01 --weight_adv 0.1 --weight_grad_adv 0.001\
 --n_iters 4 --loss l1\
 --pretrained_style  None\
 --pretrained  /home/anna/style_results/FullModel2021-09-27_styleFull_batch/`

## Citation

If you use this package in your research, please cite the following paper:

[Signal Domain Learning Approach for Optoacoustic Image Reconstruction from Limited View Data](https://openreview.net/forum?id=9NOyrfUBtx1)

## Acknowledgements

This project is supported by [Swiss Data Science Center (SDSC)](https://datascience.ch/) grant C19-04.

## License

This project is licensed under [MIT License](https://mit-license.org/).