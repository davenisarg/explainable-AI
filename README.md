# [Explainable AI for Understanding Decisions & Data-Driven Optimization in Deep Neural Networks](https://www.researchgate.net/publication/326586154_Explainable_AI_for_Understanding_Decisions_and_Data-Driven_Optimization_of_the_Choquet_Integral)

## PART 1: Building [Deep Neural Networks](https://en.wikipedia.org/wiki/Deep_learning#Deep_neural_networks) on [ImageNet](http://www.image-net.org/) 
Repository contains code for training different architectures of image classification (i.e. [GoogleNet](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf), [ResNet-50 & 101](https://arxiv.org/abs/1512.03385), [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) etc.) on ImageNet(Orginal) dataset.

**************************
**Features:**
1. The code reads dataset information from a text or csv file and directly loads images from disk.
2. Code Utilizes the __GPU parellelization__ during training phase.
3. Code is built for running on __[High-Performance Computing(Super Computing) Infrastructure](https://en.wikipedia.org/wiki/Supercomputer)__.
4. Completely designed, by performing all necessary validation checks.
5. Compatible with both Python 3.5.x and Python 3.6.x (tested)
6. Code has personalization for selecting optimization algorithm, learning rate and weight decay policies\
7. Code can build various architectures in optimized manner
8. Code Supports full automation for training, validation and testing phase.
**************************

**************************
**Required Modules/Packages/libraries:**\
__Python 3.7__ : os, sys, datetime, time, future, argparse, numpy, tensorflow
**************************

**************************
**Required Scripts/files:**\
__Data files__           : /path-to/dataset/train_imagenet and /path-to/data/val_imagenet, /path-to/train.txt and /path-to/val.txt \
__Main source code file__: run.py \
__Dependent code files__ : data_loader.py, utils.py, common.py, model.py, googlenet.py, resnet.py
**************************

## Usage: 

To start, __train.txt__ file is needed. it looks something like this,

train_imagenet/n01440764/n01440764_7173.JPEG,0\
train_imagenet/n01440765/n01440764_3724.JPEG,0\
train_imagenet/n01440766/n01440764_7719.JPEG,0\
train_imagenet/n01440767/n01440764_7304.JPEG,0\
train_imagenet/n01440768/n01440764_8469.JPEG,0

Use the --delimiter option to specify the delimiter character, and --path_prefix to add a constant prefix to all the paths.

## Model Training:

For training __GoogleNet__ execute run.py with given command,

```bash
python run.py train --architecture googlenet --path_prefix ${HOME}/path-to-dataset-folder --train_info train.txt --optimizer adam --num_epochs 5
```
For training __ResNet-50__ & __ResNet-101__ execute run.py with given command,

```bash
python run.py train --architecture resnet --path_prefix ${HOME}/path-to-dataset-folder --train_info train.txt --optimizer adam --num_epochs 5 --depth 50
```
```bash
python run.py train --architecture resnet --path_prefix ${HOME}/path-to-dataset-folder --train_info train.txt --optimizer adam --num_epochs 5 --depth 101
```

For training __Alexnet(CaffeNet)__ execute run.py with given command,

```bash
python run.py train --architecture alexnet --path_prefix ${HOME}/path-to-dataset-folder --train_info train.txt --optimizer adam --num_epochs 5
```

## Model Validation:

For validation, execute run.py with given command,

```bash
python run.py eval --architecture googlenet --log_dir "googlenet_Run-02-12-2018-15:40:00" --path_prefix /path/to/imagenet/train/ --val_info val.txt
```
## Model Testing (Making Predictions):

For testing, execute run.py with given command,

```bash
python run.py inference --architecture googlenet --log_dir "googlenet_Run-02-12-2018-15:40:00" --path_prefix /path/to/imagenet/train/ --val_info val.txt --save_predictions preds.txt
```

**************************
**Customization options:**\
__1.  Deep neural networks__   : (option)--architecture --> (possible values)  googlenet and resnet \
__2.  Execution methods__      : train (for training), eval (for validating), inference (for testing) \
__3.  dataset path prefix__    : (option)--path_prefix  --> ${HOME}/path-to-dataset-folder \
__4.  Train/validation info__  : (option)--train_info train.txt and (option)--val_info val.txt \
__5.  Optimizer for DNN__      : (option)--optimizer --> (momentum(default), adam, adadelta, adagrad, rmsprop, sgd) \
__6.  learning rate policy__   : (option)--policy_type --> (constant, piecewise_linear(default), exponential) \
__7.  LR Change detials__      : (option)--LR_details --> (19, 30, 44, 53, 0.01, 0.005, 0.001, 0.0005, 0.0001)(default)\
__8.  GPU numbers(Training)__  : (option)--num_gpus --> default is 1, INTEGER(N --> 5,10,50,...) \
__9.  epoch for training__     : (option)--num_epochs --> 5,10,50,10,200 \
__10. Depth for ResNet__       : (option)--depth --> default is 50 (can change to 50,101)\
__11. Log(validation/testing)__: (option)--log_dir --> "googlenet_Run-02-12-2018-15:40:00"\
__12. Save prections__         : (option)--save_predictions --> "predictions.csv" (default) (can specify other file name). Save top-n predictions of the networks along with their confidence in the specified file\
__13. Weight decay policy__    : (option)--WD_policy --> (constant, piecewise_linear(default), exponential) \
__14. WD change details__      : (option)--WD_details --> (30, 0.0005, 0.0)(default)\
__15. Batch size__             : (option)--batch_size --> 128 (default) (can specify other value)\
__16. No of Prefectch Images__ : (option)--num_prefetch --> 2000 (default) (can specify other value)\
__17. Shuffle training data__  : (option)--shuffle --> TRUE (default) (can change it to false)\
__18. Top N accuracy__         : (option)--top_n --> 5(default) (specify top n accuracy number) \
__19. Debugging log__          : (option)--log_dir --> NONE (default) (can specify Path for saving debugging info & checkpoints)\
__20. Log runtime & mem usage__: (option)--log_debug_info --> False(default) (can be TRUE)\
__21. Maximum snapshot__       : (option)--max_to_keep --> 5(default) (Specify Maximum number of snapshot files to keep)
**************************

**************************
**Scripts:**\
__1. run.py__          : Main python script for DNN, A program to apply different well-known deep learning architectures.Ties all scripts together & performs training, validation,& testing of DNN using all scripts/functions\
__2. data_loader.py__  : Performs data loading using given text files and prepares data for model training. \
__3. utils.py__        : Utility class for computing averages of loss and accuracies, getting batches for each epoch. Determines learning rate policy and optimization algorithm type as per arguments.\
__4. common.py__       : helper function file for each model training (Contains functions/methods for batch normalization,flatten, max pool, avg pool, fully connected, spatial Convolution etc.)\
__5. model.py__        : Helper file with necessary methods/functions for simulating model building, training & validation.\
__6. alexnet.py__      : Alexnet(Caffenet) Implementation (Deep Convolution Network architecture)\
__7. googlenet.py__    : GoogleNet Implementation (Deep Convolution Network architecture) \
__8. resnet.py__       : ResNet-50 and ResNet-101 Implementation (Deep Convolution Network architecture)
**************************

## PART 2: Investigating various XAI methods (Future Work- Coming soon)
