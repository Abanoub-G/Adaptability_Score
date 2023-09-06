import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import datetime
import copy
import torch.optim.lr_scheduler as lr_scheduler

import torch
from torch import nn, optim
from torchvision import datasets, transforms

from utils.common import set_random_seeds, set_cuda, logs
from utils.dataloaders import pytorch_dataloader, cifar10_c_dataloader, samples_dataloader

from utils.model import model_selection

from metrics.accuracy.topAccuracy import top1Accuracy

from methods.EWC import on_task_update, train_model_ewc

import matplotlib.pyplot as plt
from math import log

# =====================================================
# == Declarations
# =====================================================
SEED_NUMBER              = 0
USE_CUDA                 = True

DATASET_DIR              = '../datasets/CIFAR10/'
DATASET_NAME             = "CIFAR10" # Options: "CIFAR10" "CIFAR100" "TinyImageNet"  "ImageNet"
NUM_CLASSES              = 1000 # Number of classes in dataset

MODEL_CHOICE             = "resnet" # Option:"resnet" "vgg"
MODEL_VARIANT            = "resnet18" # Common Options: "resnet18" "vgg11" For more options explore files in models to find the different options.

MODEL_DIR                = "../models/" + MODEL_CHOICE
MODEL_SELECTION_FLAG     = 2 # create an untrained model = 0, start from a pytorch trained model = 1, start from a previously saved local model = 2

MODEL_FILENAME     = MODEL_VARIANT +"_"+DATASET_NAME+".pt"
MODEL_FILEPATH     = os.path.join(MODEL_DIR, MODEL_FILENAME)


NOISE_TYPES_ARRAY = ["brightness","motion_blur","contrast"]#,"defocus_blur",
					# "elastic_transform","fog","frost","gaussian_blur",
					# "gaussian_noise", "glass_blur", "impulse_noise",
					# "jpeg_compression", "motion_blur", "pixelate", 
					# "saturate", "shot_noise", "snow", "spatter", 
					# "speckle_noise", "zoom_blur"]

NOISE_SEVERITY 	  = 5 # Options from 1 to 5

MAX_SAMPLES_NUMBER = 200

def main():

	# ========================================
	# == Preliminaries
	# ========================================
	# Fix seeds to allow for repeatable results 
	set_random_seeds(SEED_NUMBER)

	# Setup device used for training either gpu or cpu
	device = set_cuda(USE_CUDA)

	# Load model
	model = model_selection(model_selection_flag=MODEL_SELECTION_FLAG, model_dir=MODEL_DIR, model_choice=MODEL_CHOICE, model_variant=MODEL_VARIANT, saved_model_filepath=MODEL_FILEPATH, num_classes=NUM_CLASSES, device=device)
	print("Progress: Model has been setup.")

	# Setup original dataset
	trainloader, testloader = pytorch_dataloader(dataset_name=DATASET_NAME, dataset_dir=DATASET_DIR, images_size=32, batch_size=64)
	print("Progress: Dataset Loaded.")

	# accuracies = []

	_,eval_accuracy     = top1Accuracy(model=model, test_loader=testloader, device=device, criterion=None)
	print("Model Accuray on original dataset = ",eval_accuracy)

	# Initiate dictionaries for regularisation using EWC	
	fisher_dict = {}
	optpar_dict = {}

	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
	fisher_dict, optpar_dict = on_task_update(0, trainloader, model, optimizer, fisher_dict, optpar_dict, device)
	print("PROGRESS: Calculated Fisher matrix")

	# ========================================
	# == Load Noisy Data
	# ========================================
	logs_dic = {}
	results_log = logs() 

	for noise_type in NOISE_TYPES_ARRAY:
		if noise_type == "original":
			testloader_c = testloader
		else:
			# load noisy dataset
			testloader_c, noisy_images, noisy_labels    = cifar10_c_dataloader(NOISE_SEVERITY, noise_type)
			# print("shape of noisy images = ", np.shape(noisy_images))
			# print("shape of noisy labels = ", np.shape(noisy_labels))
		
		# Evaluate testloader_c on original model
		_,eval_accuracy     = top1Accuracy(model=model, test_loader=testloader_c, device=device, criterion=None)
		original_model_eval_accuracy       = eval_accuracy.cpu().numpy()
		print(noise_type +" dataset = ",original_model_eval_accuracy)
		

		# ========================================	
		# == Select random N_T number of datapoints from noisy data for retraining
		# ========================================
		N_T_vs_A_T = []
		# Extract N_T random samples
		for N_T in range(2,MAX_SAMPLES_NUMBER,16):
			print("++++++++++++++")
			print("N_T = ", N_T)
			print("Noise Type = ", noise_type) 
			N_T_testloader_c = samples_dataloader(N_T, noisy_images, noisy_labels)


			# ========================================	
			# == Retrain model
			# ========================================

			
			
			# # Set Thresholds for controlled forgetting
			# CF_lim = 80  # Threshold for controlled forgetting
			# zeta = 1    # Constant selected to show the importance of not dropping below the CF_lim 
			num_retrain_epochs = 10
			CF_lim = 0.9

			temp_list_retrained_models = []
			temp_list_lr               = []
			temp_list_lambda           = []
			temp_list_CFAS             = []

			for lr_retrain in [1e-6]:#1e-5]:#,1e-3,1e-2]:
				for lambda_retrain in [1]:#[1,20,50,80,100]:

					# Copy model for retraining
					retrained_model = copy.deepcopy(model)
					
					# Retrain
					retrained_model = train_model_ewc(model = retrained_model, 
													train_loader = N_T_testloader_c, 
													test_loader = N_T_testloader_c, 
													device = device, 
													fisher_dict = fisher_dict,
													optpar_dict = optpar_dict,
													num_epochs=num_retrain_epochs, 
													ewc_lambda = lambda_retrain,
													learning_rate=lr_retrain, 
													momentum=0.9, 
													weight_decay=1e-5)


					# ========================================	
					# == Evaluate Retrained model
					# ========================================
					
					# Calculate accuracy of retrained model on target dataset X_tar 
					_, A_0    = top1Accuracy(model=retrained_model, test_loader=testloader, device=device, criterion=None)
					print("A_0 = ",A_0)

					# Calculate accuracy of retrained model on target dataset X_tar 
					_, A_k    = top1Accuracy(model=retrained_model, test_loader=N_T_testloader_c, device=device, criterion=None)
					print("A_k = ",A_k)

					# Calculate CFAS
					CFAS = (A_0.cpu().numpy() - CF_lim)+(A_k.cpu().numpy() - 1)

					# Append Data
					temp_list_retrained_models.append(retrained_model)
					temp_list_lr.append(lr_retrain)
					temp_list_lambda.append(lambda_retrain)
					temp_list_CFAS.append(CFAS)     

			index_max = np.argmax(temp_list_CFAS)
			retrained_model = temp_list_retrained_models[index_max]
			CFAS = temp_list_CFAS[index_max]
			# lr = temp_list_lr[index_max]
			# lambda = temp_list_lambda[index_max]

			# Calculate accuracy of retrained model on target dataset X_tar 
			_, A_T    = top1Accuracy(model=retrained_model, test_loader=testloader_c, device=device, criterion=None)
			print("A_T = ",A_T)

			# best = fmin(fn=lambda x: retraining_objective(x),
			# 			space= {'x': [hp.uniform('ewc_lambda_hyper', 0, 100), hp.uniform('lr_retrain_hyper', 1e-5, 1e-2)]},
			# 			algo=tpe.suggest,
			# 			max_evals=100)

			# while True:
			# 	# # Calculate accuracy of retrained model on original dataset X_0 
			# 	# _, A_0    = top1Accuracy(model=retrained_model, test_loader=testloader, device=device, criterion=None)
				
			# 	# # Calculate accuracy of retrained model on samples, N_T, from target dataset X_tar 
			# 	# _, A_k    = top1Accuracy(model=retrained_model, test_loader=N_T_testloader_c, device=device, criterion=None)
				
			# 	# # Calculate CFAS for retrained model
			# 	# CFAS = zeta * (CF_lim - A_0) + (100 - A_k)
				
			# 	# # If CFAS satisfied, break while loop


			# 	# # Else update parameters

			# 	# Retrain
			# 	retrained_model = retrain(model, testloader, N_T_testloader_c)

			# # ========================================	
			# # == Evaluate Retrained model
			# # ========================================
			
			# # print("A_0 = ",original_model_eval_accuracy)
			
			# # Calculate accuracy of retrained model on target dataset X_tar 
			# _, A_0    = top1Accuracy(model=retrained_model, test_loader=testloader, device=device, criterion=None)
			# print("A_0 = ",A_0)

			# # Calculate accuracy of retrained model on target dataset X_tar 
			# _, A_T    = top1Accuracy(model=retrained_model, test_loader=testloader_c, device=device, criterion=None)
			# print("A_T = ",A_T)
			# # Evaluate testloader on original model 
			# _,eval_accuracy     = top1Accuracy(model=retrained_model, test_loader=testloader, device=device, criterion=None)
			# print("Model Accuray on original dataset = ",eval_accuracy)
			
			# # Evaluate testloader_c on original model
			# _,eval_accuracy     = top1Accuracy(model=retrained_model, test_loader=testloader_c, device=device, criterion=None)
			# print("Model Accuray on noisy dataset = ",eval_accuracy)
			
			# # Evaluate N_T_testloader_c on original model
			# _,eval_accuracy     = top1Accuracy(model=retrained_model, test_loader=N_T_testloader_c, device=device, criterion=None)
			# print("Model Accuray on noisy dataset = ",eval_accuracy)

			N_T_vs_A_T.append((N_T, A_T))
			results_log.append(noise_type, N_T, A_T)
		
		# Log
		logs_dic[noise_type] = N_T_vs_A_T

	results_log.write_file("exp3.txt")
	# ========================================	
	# == Plot
	# ========================================
	# for noise_type in NOISE_TYPES_ARRAY:
	# 	# plot_list = logs_dic[noise_type]
	# 	# plot_list2 = [(elem1, log(elem2)) for elem1, elem2 in plot_list]
	# 	# zip(*plot_list2)
	# 	# plt.scatter(*zip(*plot_list2))
	# 	plt.scatter()
	# plt.savefig('foo.png')

	# input("Press enter")

		# Retrain model: using SGD and using SGD+EWC

		# Evaluate testloader on retrained model 
		# Evaluate testloader_c on retrained model 
		# Evaluate N_T_testloader_c on retrained model




		# # ========================================	
		# # == Evaluate Retrained model
		# # ========================================


		# _,eval_accuracy     = top1Accuracy(model=model, test_loader=testloader_c, device=device, criterion=None)
		# print("Model Accuray on "+noise_type+" dataset = ",eval_accuracy)
		
		# eval_accuracy = eval_accuracy.cpu().numpy()

		# accuracies.append(eval_accuracy)

		# print("========")
		
		# model.eval()
		# model.to(device)

		# for inputs, labels in testloader_c:
		# 	# Predict
		# 	inputs = inputs.to(device)
		# 	labels = labels.to(device)
			
		# 	outputs = model(inputs)
		# 	_, preds = torch.max(outputs, 1)








if __name__ == "__main__":

	main()