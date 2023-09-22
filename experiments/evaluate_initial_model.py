import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import datetime
import torch.optim.lr_scheduler as lr_scheduler

import torch
from torch import nn, optim
from torchvision import datasets, transforms

from utils.common import set_random_seeds, set_cuda
from utils.dataloaders import cifar_c_dataloader


from utils.model import model_selection, train_model, save_model
from utils.dataloaders import pytorch_dataloader
from metrics.accuracy.topAccuracy import top1Accuracy


# =====================================================
# == Declarations
# =====================================================
SEED_NUMBER              = 0
USE_CUDA                 = True

DATASET_DIR              = '../datasets/CIFAR100/'
DATASET_NAME             = "CIFAR100" # Options: "CIFAR10" "CIFAR100" "TinyImageNet"  "ImageNet"
NUM_CLASSES              = 1000 # Number of classes in dataset

MODEL_CHOICE             = "resnet" # Option:"resnet" "vgg"
MODEL_VARIANT            = "resnet26" # Common Options: "resnet18" "vgg11" For more options explore files in models to find the different options.

MODEL_DIR                = "../models/" + MODEL_CHOICE
MODEL_SELECTION_FLAG     = 2 # create an untrained model = 0, start from a pytorch trained model = 1, start from a previously saved local model = 2

SAVED_MODEL_FILENAME     = MODEL_VARIANT +"_"+DATASET_NAME+".pt"
SAVED_MODEL_FILEPATH     = os.path.join(MODEL_DIR, SAVED_MODEL_FILENAME)

TRAINED_MODEL_FILENAME   = MODEL_VARIANT +"_"+DATASET_NAME+".pt"

NUM_EPOCHS               = 10
LEARNING_RATE            = 1e-5

NOISE_TYPES_ARRAY = ["original","brightness","contrast","defocus_blur",
                    "elastic_transform","fog","frost","gaussian_blur",
                    "gaussian_noise", "glass_blur", "impulse_noise",
                    "jpeg_compression", "motion_blur", "pixelate", 
                    "saturate", "shot_noise", "snow", "spatter", 
                    "speckle_noise", "zoom_blur"]

NOISE_SEVERITY = 5
# Fix seeds to allow for repeatable results 
set_random_seeds(SEED_NUMBER)

# Setup device used for training either gpu or cpu
device = set_cuda(USE_CUDA)


def main():
    # Fix seeds to allow for repeatable results 
    set_random_seeds(SEED_NUMBER)

    # Setup device used for training either gpu or cpu
    device = set_cuda(USE_CUDA)


    # Setup model
    model = model_selection(model_selection_flag=MODEL_SELECTION_FLAG, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=SAVED_MODEL_FILEPATH, num_classes=NUM_CLASSES, device=device)
    print("Progress: Model has been setup.")

    # Setup dataset
    eval_accuracy_array = []
    for noise_type in NOISE_TYPES_ARRAY:
        if noise_type == "original":
            _, testloader = pytorch_dataloader(dataset_name=DATASET_NAME, dataset_dir=DATASET_DIR, images_size=32, batch_size=64)
        else:
            # load noisy dataset
            _,testloader, _, _    = cifar_c_dataloader(NOISE_SEVERITY, noise_type, DATASET_NAME)
        
        print("Progress: Dataset Loaded.")


        # Evaluate model
        _,eval_accuracy     = top1Accuracy(model=model, test_loader=testloader, device=device, criterion=None)
        eval_accuracy_array.append(eval_accuracy)
        print(noise_type+ " = "+str(eval_accuracy))

    plt.clf()
    fig = plt.figure()
    plt.rcParams.update({'font.size': 2})
    ax = fig.add_axes([0.15,0.14,0.8,0.8])
    ax.bar(NOISE_TYPES_ARRAY, eval_accuracy_array, color = 'k', width = 0.25)

    plt.ylabel("Evaluation Accuracy")
    plt.savefig("AccuracyVSnoise_"+DATASET_NAME+"_"+MODEL_VARIANT+".pdf")

   
    # print("FP32 evaluation accuracy: {:.3f}".format(eval_accuracy))
    
if __name__ == "__main__":

    main()
