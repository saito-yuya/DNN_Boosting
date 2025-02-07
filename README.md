# DNN_Boosting

This repository provides the code for my research [DNN×Boosting] in Pytorch.

[Paper] | [Bibtex] | [Slides]

## Overveiw of Our Method

![Illustration](./images/overview.png)
> This paper tackles the problem of the worst-class error rate, instead of the standard error rate averaged over all classes. For example, a three-class classification task with class-wise error rates of 10%, 10%, and 40% has a worst-class error rate of 40%, whereas the average is 20% under the class-balanced condition.
The worst-class error is important in many applications. For example, in a medical image classification task, it would not be acceptable for the malignant tumor class to have a 40\% error rate, while the benign and healthy classes have a 10\% error rates. To avoid overfitting in worst-class error minimization using Deep Neural Networks (DNNs), we design a problem formulation for bounding the worst-class error instead of achieving zero worst-class error. Moreover, to correctly bound the worst-class error, we propose a boosting approach which ensembles DNNs.
We give training and generalization worst-class-error bound.
Experimental results show that the algorithm lowers worst-class test error rates while avoiding overfitting to the training set.

## Requirements 
<!-- All codes are written by Python 3.7, and 'requirements.txt' contains required Python packages. -->
- python >= 3.8
- cuda & cudnn

### prerequisitions
- python 3.8.17
- seaborn  0.12.2
- scikit-learn  1.3.0
- pandas 2.0.3
- Pillow 10.0.0
- torch  2.0.1
- torchvision 0.15.2
- pytorch_transformers
- tqdm  4.65.0
- opencv-python 4.8.0.74
- medmnist 2.2.2
- numpy 1.22.3


To install fast-setup of this code:

```setup
# pytorch install 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```



## Dataset

Create a ```data/``` directory and download the original data into this directory to generate imbalanced versions.
- Imbalanced [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html). The original data will be downloaded and converted by `imbalancec_cifar.py`.
- The paper also reports results on [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip), [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) . We will update the code for Tiny ImageNet & EMNIST later.

## Training & Test

We provide several examples:

### CIFAR10
- CE baseline

```bash
python cifar_train.py --dataset cifar10 -a resnet32 --loss_type CE --train_rule None --epochs 200 --b 512 --num_classes 10 --gpu 0 --early_stop True --stop_mode average
```
- IB + CB

```bash
python cifar_train.py --dataset cifar10 -a resnet32 --loss_type IB --train_rule CBReweight --epochs 200 --b 512 --start_ib_epoch 50 --num_classes 10 --gpu 0 --early_stop True --stop_mode average 
```
- Naive

```bash
python cifar_train.py --dataset cifar10 -a resnet32 --loss_type WorstLoss --train_rule None --epochs 200 --b 512 --num_classes 10 --gpu 0 --early_stop True --stop_mode worst
```

- VS

```bash
python cifar_train.py --dataset cifar10 -a resnet32 --loss_type VS --train_rule None --epochs 200 --b 512 --num_classes 10 --gpu 0 --early_stop True --gamma 0.15 --tau 1.25 --stop_mode average
```

- LA

```bash
python cifar_train.py --dataset cifar10 -a resnet32 --loss_type LA --train_rule None --epochs 200 --b 512 --num_classes 10 --gpu 0 --early_stop True --gamma 0.0 --tau 2.25 --stop_mode average
```

- Ours
  
```bash
python ./Ours/cifar_train.py --dataset cifar10 -a resnet32 --theta 0.9 --loss_type CE --b 512 --num_classes 10 --gpu 0 

```


### Medmnist (TisuueMNIST)
Fast of all, changes the code of ```models/__init__.py``` and run the listed example code.

```bash
# from .resnet_cifar import * 
from .resnet_med_resnet_s import *
```

- CE (w /fCW)

```bash
python medmnist_train.py --dataset medmnist --data_flag tissuemnist -a resnet18 --num_in_channels 1 --loss_type CE --train_rule fCW --epochs 100 --b 512 --num_classes 8 --gpu 0 --early_stop True --stop_mode average
```
- Focal

```bash
python medmnist_train.py --dataset medmnist --data_flag tissuemnist -a resnet18 --num_in_channels 1 --loss_type Focal --epochs 100 --b 512 --num_classes 8 --gpu 0 --early_stop True --stop_mode average 
```

- CDT

```bash
python medmnist_train.py --dataset medmnist --data_flag tissuemnist -a resnet18 --num_in_channels 1 --loss_type CDT --epochs 100 --b 512 --num_classes 8 --gpu 0 --early_stop True --gamma 0.4 --tau 0.0 --stop_mode average 
```

- Ours

```bash
python ./Ours/medmnist_train.py --dataset medmnist --data_flag tissuemnist -a resnet18 --theta 0.6 --num_in_channels 1 --b 512 --num_classes 8 --loss_type CE --gpu 0 
```


```

```