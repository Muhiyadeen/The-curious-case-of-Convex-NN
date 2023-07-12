* cifar10_train_codes folder contains code for training three architectures presented in the paper namely, MLP, AllConv and Densenet. 

* run_net.py takes two arguments : 1) model name (either mlp, conv or densenet) and 2) model type (ioc or nn). It creates a directory inside cifar10 with a name corresponding to model name and type (For example: mlp_ioc), inside which the log file is saved. 

* cifar10_networks.py is a helper file. This file is called by run_net.py at execution.

* requirement.txt: lists libraries and versions required to set the environment to run these codes.
