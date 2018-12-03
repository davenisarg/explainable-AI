# Explainable AI for Understanding Decisions & Data-Driven Optimization in Deep Neural Networks

## Building Deep Neural Networks on ImageNet
Repository contains code for training different architectures of image classification (i.e. GoogleNet, ResNet, AlexNet etc.) on ImageNet dataset.

**************************
**Features**\
The code reads dataset information from a text or csv file and directly loads images from disk.\
Code Utilizes the GPU parellelization during training phase.\
Code is built for running on HPC Infrastructure.\
Completely designed, by performing all necessary validation checks.\
Compatible with both Python 2.7 and Python 3.6\
Code has personalization for selecting optimization algorithm, learning rate and weight decay policies\
Code can build various architectures in optimized manner\
Code Supports full automation for training, validation and testing phase.
**************************

**************************
**Required Modules/Packages/libraries**\
Python 3.7 : os, sys, datetime, time, __future__, argparse, numpy, tensorflow
**************************

**************************
**Required Scripts/files**\
Data files           : /path-to/dataset/train_imagenet and /path-to/data/val_imagenet, /path-to/train.txt and /path-to/val.txt \
Main source code file: run.py \
Dependent code files : data_loader.py, utils.py, common.py, model.py, googlenet.py, resnet.py
**************************

# Usage:

To start, train.txt file is needed. it looks something like this,

train_imagenet/n01440764/n01440764_7173.JPEG,0\
train_imagenet/n01440765/n01440764_3724.JPEG,0\
train_imagenet/n01440766/n01440764_7719.JPEG,0\
train_imagenet/n01440767/n01440764_7304.JPEG,0\
train_imagenet/n01440768/n01440764_8469.JPEG,0

Use the --delimiter option to specify the delimiter character, and --path_prefix to add a constant prefix to all the paths.

For training execute run.py with given command,
```bash
python run.py train --architecture googlenet --path_prefix ${HOME}/path-to-dataset-folder --train_info train.txt --optimizer adam --num_epochs 5 --num_gpus 25
```

For validation execute run.py with given command,
```bash
python run.py eval --architecture googlenet --log_dir "googlenet_Run-02-12-2018-15:40:00" --path_prefix /path/to/imagenet/train/ --val_info val.txt
```

For testing execute run.py with given command,
```bash
python run.py inference --architecture googlenet --log_dir "googlenet_Run-02-12-2018-15:40:00" --path_prefix /path/to/imagenet/train/ --val_info val.txt --save_predictions preds.txt
```
**************************
**Customization options**\
1. Deep neural networks   : (option)--architecture --> (possible values)  googlenet and resnet \
2. Execution methods      : train (for training), eval (for validating), inference (for testing) \
3. dataset path prefix    : (option)--path_prefix  --> ${HOME}/path-to-dataset-folder \
4. Train/validation info  : (option)--train_info train.txt and (option)--val_info val.txt \
5. Optimizer for DNN      : (option)--optimizer --> (momentum, adam, adadelta, adagrad, rmsprop, sgd) \
6. learning rate policy   : (option)--policy_type --> (constant, piecewise_linear, exponential) \
7. GPU numbers(Training)  : (option)--num_gpus --> default is 1, INTEGER(N) \
8. epoch for training     : (option)--num_epochs --> 5,10,50,10,200 \
9. Log(validation/testing): (option)--log_dir --> "googlenet_Run-02-12-2018-15:40:00"
**************************
