# Convergence Analysis of Mini-Batch Stochastic Gradient Descent Using Linearand Cosine-annealing Learning-rate Schedulers
# Abstract
Mini-batch stochastic gradient descent (SGD) is a power
ful deep-learning optimizer for finding appropriate param
eters of a deep neural network in the sense of minimizing
the empirical loss defined by the mean of the loss func
tions corresponding to the training set. The performance of
mini-batch SGD strongly depends on the setting of the learn
ing rate, and various learning rates have been used in prac
tice. In particular, linear and cosine-annealing learning-rate
schedulers are considered be practical schedulers with which
deep-learning optimizers perform well. In this paper, we pro
vide convergence analyses of mini-batch SGD using linear
and cosine-annealing learning rates. The analyses indicate
that using a linear or cosine-annealing learning rate is bet
ter than using a constant learning rate in the sense of mini
mizing the expectation of the full gradient norm of the em
pirical loss. We also provide numerical results showing that
linear and cosine-annealing learning rates are especially use
ful for training deep neural networks.

# Usage
please change method, lr, batchsize, epochs, models(ResNet18, WideResNet28-10), warmup_epochs or steps when train model by CIFAR100

```python cifar100.py --method constant --lr 0.01 --model ResNet18 --warmup_epochs 0 --batchsize 64 --epochs 200```

