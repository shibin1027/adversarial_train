# Adversarial Training Tools

## Train

We have adversarial training script for AT(baseline), SP on CIFAR10, CIFAR100, TinyImageNet dataset. This repo can served as a general training pipeline for further reference.

We will evaluate the robustness of the trained model every epoch, and save the current model and the best model. This will take extra training time, but avoid overfitting.

## Evaluate

We evaluate the model using several basic attack methods, such as FGSM, PGD, CW. 

> All these script are firstly used for the work of VMF.