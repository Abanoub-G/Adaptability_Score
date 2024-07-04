# Dynamic Incremental Regularised Adaptation (DIRA)

This is the official repository for our paper: [*DIRA: Dynamic Incremental Regularised Adaptation*](https://arxiv.org/abs/2205.00147)

Autonomous systems (AS) often use Deep Neural Network (DNN) classifiers to allow them to operate in complex, high-dimensional, non-linear, and dynamically changing environments. Due to the complexity of these environments, DNN classifiers may output misclassifications during operation when they face domains not identified during development. Removing a system from operation for retraining becomes impractical as the number of such AS increases. To increase AS reliability and overcome this limitation, DNN classifiers need to have the ability to adapt during operation when faced with different operational domains using a few samples (e.g. 100 samples). However, retraining DNNs on a few samples is known to cause catastrophic forgetting. In this paper, we introduce Dynamic Incremental Regularised Adaptation (DIRA), a framework for operational domain adaption of DNN classifiers using regularisation techniques to overcome catastrophic forgetting and achieve adaptation when retraining using a few samples of the target domain. Our approach shows improvements on different image classification benchmarks aimed at evaluating robustness to distribution shifts (e.g.CIFAR-10C/100C, ImageNet-C), and produces state-of-the-art performance in comparison with other frameworks from the literature.

# Installation
1) Clone this repository: `git clone` 
2) Clone our docker image and setup container: `docker run -t --runtime=nvidia --shm-size 16G -d --name netzip -v ~/gits:/home -p 5000:80 abanoubg/netzip:latest`.


# Setup Datasets

# Quick Start
