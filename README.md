# CornealAI
This is python code for implementation in the paper: 
>Nicole Hallett, Kai Yi, Josef Dick, Christopher Hodge, Gerard Sutton, Yu Guang Wang, Jingjing You[Deep Learning Based Unsupervised and Semi-supervised Classification for Keratoconus](https://arxiv.org/abs/2001.11653). arXiv preprint arXiv:2001.11653, 2020.

## Abstract
The transparent cornea is the window of the eye,
facilitating the entry of light rays and controlling focusing the
movement of the light within the eye. The cornea is critical, contributing to 75% of the refractive power of the eye. Keratoconus
is a progressive and multifactorial corneal degenerative disease
affecting 1 in 2000 individuals worldwide. Currently, there is
no cure for keratoconus other than corneal transplantation for
advanced stage keratoconus or corneal cross-linking, which can
only halt KC progression. The ability to accurately identify subtle
KC or KC progression is of vital clinical significance. To date,
there has been little consensus on a useful model to classify KC
patients, which therefore inhibits the ability to predict disease
progression accurately.

In this paper, we utilised machine learning to analyse data from
124 KC patients, including topographical and clinical variables.
Both supervised multilayer perceptron and unsupervised variational autoencoder models were used to classify KC patients with
reference to the existing Amsler-Krumeich (A-K) classification
system. Both methods result in high accuracy, with the unsupervised method showing better performance. The result showed that
the unsupervised method with a selection of 29 variables could
be a powerful tool to provide an automatic classification tool
for clinicians. These outcomes provide a platform for additional
analysis for the progression and treatment of keratoconus.


## Classification
For training, one can run 
```
python train.py
```
There are some parameters for the train program.
```
TRAIN_DIR               -  the path training data
TEST_DIR                -  the path test data
model_checkpoint_dir    -  model weights save directory   
epoch                   - how many epochs in training

```

## Cluster
For cluster, one can run 
```
python cluster.py
```
There are some parameters for the cluster program.
```
normalize_constant    -  the normalize_constant, default = 50
weight                -  the pretrained weight file name, if no pretrained weight, then ignore
epoch                 -  how many epochs in training

```
For example, you can run
```
python cluster.py --weight '2weights.09-2.05.h5' \
                  --normalize_constant 40 \
                  --epoch 100
```


## Cite
Please cite our paper if you use this code in your own work:
```
@article{hallett2020deep,
  title={Deep Learning Based Unsupervised and Semi-supervised Classification for Keratoconus},
  author={Hallett, Nicole and Yi, Kai and Dick, Josef and Hodge, Christopher and Sutton, Gerard and Wang, Yu Guang and You, Jingjing},
  journal={arXiv preprint arXiv:2001.11653},
  year={2020}
}
```
