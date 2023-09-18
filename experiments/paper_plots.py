import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

# =======================================
# Samples VS Accuracy
# =======================================
experiment_id_array = [66, 64, 45, 41]

Source_acuracy = 0.637#0.504

for experiment_id in experiment_id_array:

	file_name = "Results_logs/exp"+str(experiment_id)+".txt"
	results_folder = "Results_plots"

	df = pd.read_csv(file_name)
	NOISE_TYPES = np.unique(df["noise_type"])

	first_run_flag = True
	for noise_type in NOISE_TYPES:
		df_extracted  = df[df.loc[:,"noise_type"]==noise_type]
		N_T_array = df_extracted["N_T"].values

		if first_run_flag:
			A_T_array = df_extracted["A_T"].values
			first_run_flag = False
		else:
			A_T_array += df_extracted["A_T"].values

	A_T_array = A_T_array/len(NOISE_TYPES)
	A_T_array = np.insert(A_T_array,0, Source_acuracy)
	N_T_array = np.insert(N_T_array,0, 0)

	if experiment_id == 66:
		label_name = "DIRA"
		color_name = "blue"
		linestyle  = "-"
	
	elif experiment_id == 41:
		label_name = "SGD ($\eta$ = 1e-5"
		color_name = "orange"
		linestyle  = "-"

	elif experiment_id == 45:
		label_name = "SGD ($\eta$ = 1e-3"
		color_name = "orange"
		linestyle  = "-."

	elif experiment_id == 64:
		label_name = "SGD ($\eta$ = 1e-2"
		color_name = "orange"
		linestyle  = "--"

	

	plt.plot(N_T_array, A_T_array*100, label=label_name, color = color_name, linestyle=linestyle)

plt.plot([0,202], [Source_acuracy*100, Source_acuracy*100], label="Source", linestyle="--", color="grey")


plt.legend()
plt.ylim(0,100)
plt.xlim(0,200)
plt.xlabel("Number of Samples for Adaptation")
plt.ylabel("Top-1 Classification Accuracy (%)")
plt.savefig(results_folder+"/CIFAR10C_Resnet18_N_TvsA_T.pdf")



# =======================================
# Dyanmic Adaptation Scenario
# =======================================
# experiment_id_array = [66, 64]


Source_acuracy = [0.270, 0.593, 0.510, 0.625] # strictly in the order of "contrast","defocus_blur", "fog", "zoom_blur"
Baseline_accuracy = 0.855

experiment_id = 66

file_name = "Results_logs/exp"+str(experiment_id)+".txt"
results_folder = "Results_plots"

df = pd.read_csv(file_name)
NOISE_TYPES = np.unique(df["noise_type"])

first_run_flag = True
last_N_T = 0

plt. clf()
plt.plot([0, 600], [Baseline_accuracy*100, Baseline_accuracy*100], label="Baseline CIFAR-10", linestyle="--", color="grey")

counter = -1
for noise_type in ["contrast","defocus_blur", "fog", "zoom_blur"]:
	counter += 1
	if noise_type == "original":
		A_T_array    = np.array([Baseline_accuracy, Baseline_accuracy])
		N_T_array    = np.array([0,100])
		

	else:

		df_extracted  = df[df.loc[:,"noise_type"]==noise_type]
		A_T_array = df_extracted["A_T"].values
		N_T_array = df_extracted["N_T"].values

		# A_T_array = A_T_array[:5] # np.insert(A_T_array[:5],0, Source_acuracy)
		# N_T_array = N_T_array[:5] - 2# np.insert(N_T_array[:5],0, 0) -2

		A_T_array = np.insert(A_T_array[:5], 0, Source_acuracy[counter])
		N_T_array = np.insert(N_T_array[:5],0, 0) 
		N_T_array[-1] = 100
	
	A_T_array = A_T_array *100

	if first_run_flag:
		pass
	else:
		# A_T_array = np.insert(A_T_array,0, A_T_array_old[-1]) 
		# N_T_array = np.insert(N_T_array,0, N_T_array_old[0])
		pass
	
	A_T_array_old    = A_T_array
	N_T_array_old    = N_T_array 

	N_T_array = N_T_array + last_N_T
	last_N_T  = N_T_array[-1]

	if first_run_flag:
		first_run_flag = False
		plt.plot(N_T_array, A_T_array, color = "blue", label="DIRA ")
		plt.plot([N_T_array[-1], N_T_array[-1]], [0,100], linestyle="--", color = "green", linewidth = '0.5', label="Change of Domain")
	else:
		plt.plot(N_T_array, A_T_array, color = "blue")
		plt.plot([N_T_array[-1], N_T_array[-1]], [0,100], linestyle="--", color = "green", linewidth = '0.5')

	if noise_type == "original":
		text = "Baseline"
		x = N_T_array[0]+15

	elif noise_type == "contrast":
		text = "Contrast"
		x = N_T_array[0]+20

	elif noise_type == "defocus_blur":
		text = "Defocus Blur"
		x = N_T_array[0]+15
	
	elif noise_type == "fog":
		text = "Fog"
		x = N_T_array[0]+35

	elif noise_type == "zoom_blur":
		text = "Zoom Blur"
		x = N_T_array[0]+20

	elif noise_type == "original":
		text = "Baseline"
		x = N_T_array[0]+10

	plt.text(x,30, text)
	plt.text(N_T_array[0]+5,25, str(round(A_T_array[0], 1))+"% -> "+str(round(A_T_array[-1], 1))+"%")#, size=7)



plt.legend()
plt.ylim(0,100)
plt.xlim(0,400)
plt.xlabel("Number of Samples")
plt.ylabel("Top-1 Classification Accuracy (%)")
plt.savefig(results_folder+"/DynamicScenario.pdf")


# ===============================================
# Adaptability Score during Adaptation Scenario
# ===============================================
# experiment_id_array = [66, 64]


Source_acuracy = [0.270, 0.593, 0.510, 0.625] # strictly in the order of "contrast","defocus_blur", "fog", "zoom_blur"
Baseline_accuracy = 0.855

experiment_id = 66

file_name = "Results_logs/exp"+str(experiment_id)+".txt"
results_folder = "Results_plots"

df = pd.read_csv(file_name)
NOISE_TYPES = np.unique(df["noise_type"])

first_run_flag = True
last_N_T = 0

plt. clf()
plt.plot([0, 600], [Baseline_accuracy*100, Baseline_accuracy*100], label="Baseline CIFAR-10", linestyle="--", color="grey")

mean_A_T = 0
mean_AS = 0

counter = -1
for noise_type in ["contrast","defocus_blur", "fog", "zoom_blur"]:
	counter+=1
	if noise_type == "original":
		A_T_array    = np.array([Baseline_accuracy, Baseline_accuracy])
		N_T_array    = np.array([0,100])
	else:
		df_extracted  = df[df.loc[:,"noise_type"]==noise_type]
		A_T_array = df_extracted["A_T"].values
		N_T_array = df_extracted["N_T"].values

		# A_T_array = A_T_array[:5] # np.insert(A_T_array[:5],0, Source_acuracy)
		# N_T_array = N_T_array[:5] - 2# np.insert(N_T_array[:5],0, 0) -2

		A_T_array = np.insert(A_T_array[:5], 0, Source_acuracy[counter])
		N_T_array = np.insert(N_T_array[:5],0, 0) 
		N_T_array[-1] = 100
	
	A_T_array = A_T_array *100

	if first_run_flag:
		pass
	else:
		# A_T_array = np.insert(A_T_array,0, A_T_array_old[-1]) 
		# N_T_array = np.insert(N_T_array,0, N_T_array_old[0])
		pass
	
	A_T_array_old    = A_T_array
	N_T_array_old    = N_T_array 

	N_T_array = N_T_array + last_N_T
	last_N_T  = N_T_array[-1]


	if first_run_flag:
		plt.plot(N_T_array, A_T_array, color = "blue", label="DIRA ")
		plt.plot([N_T_array[-1], N_T_array[-1]], [0,100], linestyle="--", color = "green", linewidth = '0.5', label="Change of Domain")
	else:
		plt.plot(N_T_array, A_T_array, color = "blue")
		plt.plot([N_T_array[-1], N_T_array[-1]], [0,100], linestyle="--", color = "green", linewidth = '0.5')

	# Find N_{T,m} and plot
	derivative_values = (A_T_array[1:] - A_T_array[:-1]) / (N_T_array[1:] - N_T_array[:-1])

	print(derivative_values)
	zeta = 0.04
	bool_array = derivative_values > zeta
	last_true_index = np.where(bool_array)[0][-1]+1

	if first_run_flag:
		first_run_flag = False
		plt.scatter(N_T_array[last_true_index], A_T_array[last_true_index], c="red", marker="o", label="$N_T$")
	else:
		plt.scatter(N_T_array[last_true_index], A_T_array[last_true_index], c="red", marker="o")


	# Calculating AS and plotting
	N_d = 100
	N_T = N_T_array[last_true_index]
	A_T = A_T_array[last_true_index]
	print(A_T)
	print(N_T)
	AS = A_T /(N_T-counter*100)

	mean_A_T += A_T_array[-1]
	mean_AS += AS




	# Write improvements and Adaptability Score 
	if noise_type == "original":
		text = "Baseline"
		x = N_T_array[0]+15

	elif noise_type == "contrast":
		text = "Contrast"
		x = N_T_array[0]+20

	elif noise_type == "defocus_blur":
		text = "Defocus Blur"
		x = N_T_array[0]+15
	
	elif noise_type == "fog":
		text = "Fog"
		x = N_T_array[0]+35

	elif noise_type == "zoom_blur":
		text = "Zoom Blur"
		x = N_T_array[0]+20

	elif noise_type == "original":
		text = "Baseline"
		x = N_T_array[0]+10

	plt.text(x,35, text)
	plt.text(N_T_array[0]+5,30, str(round(A_T_array[0], 1))+"% -> "+str(round(A_T_array[-1], 1))+"%")#, size=7)
	plt.text(N_T_array[0]+15,45, "AS =  "+str(round(AS,2)))

mean_AS = mean_AS/(counter+1)
mean_A_T = mean_A_T/(counter+1)

print("mean A_T = ", mean_A_T)
print("mean AS = ", mean_AS)

plt.legend()
plt.ylim(0,100)
plt.xlim(0,400)
plt.xlabel("Number of Samples")
plt.ylabel("Top-1 Classification Accuracy (%)")
plt.savefig(results_folder+"/AdaptabilityScore.pdf")

# ===============================================
# Adaptability Score 
# ===============================================
# Strictly in the order of ["brightness","contrast","defocus_blur","elastic_transform","fog","frost","gaussian_blur","gaussian_noise", "glass_blur", "impulse_noise","jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]
Source_acuracy = [0.783, 0.270, 0.593, 0.722, 0.510, 0.674,0.5530, 0.6200, 0.5930,0.4160,0.7866,0.6143,0.750,0.7371,0.6386, 0.7186, 0.6691, 0.6204, 0.6252] 

Baseline_accuracy = 0.855

experiment_id = 64

file_name = "Results_logs/exp"+str(experiment_id)+".txt"
results_folder = "Results_plots"

df = pd.read_csv(file_name)
NOISE_TYPES = np.unique(df["noise_type"])

first_run_flag = True
last_N_T = 0

plt. clf()
plt.plot([0, 600], [Baseline_accuracy*100, Baseline_accuracy*100], label="Baseline CIFAR-10", linestyle="--", color="grey")

mean_A_T = 0
mean_AS = 0

counter = -1
for noise_type in NOISE_TYPES:
	counter+=1
	if noise_type == "original":
		A_T_array    = np.array([Baseline_accuracy, Baseline_accuracy])
		N_T_array    = np.array([0,100])
	else:
		df_extracted  = df[df.loc[:,"noise_type"]==noise_type]
		A_T_array = df_extracted["A_T"].values
		N_T_array = df_extracted["N_T"].values

		# A_T_array = A_T_array[:5] # np.insert(A_T_array[:5],0, Source_acuracy)
		# N_T_array = N_T_array[:5] - 2# np.insert(N_T_array[:5],0, 0) -2

		A_T_array = np.insert(A_T_array[:5], 0, Source_acuracy[counter])
		N_T_array = np.insert(N_T_array[:5],0, 0) 
		N_T_array[-1] = 100
	
	A_T_array = A_T_array *100

	if first_run_flag:
		pass
	else:
		# A_T_array = np.insert(A_T_array,0, A_T_array_old[-1]) 
		# N_T_array = np.insert(N_T_array,0, N_T_array_old[0])
		pass
	
	A_T_array_old    = A_T_array
	N_T_array_old    = N_T_array 

	N_T_array = N_T_array + last_N_T
	last_N_T  = N_T_array[-1]


	if first_run_flag:
		plt.plot(N_T_array, A_T_array, color = "blue", label="DIRA ")
		plt.plot([N_T_array[-1], N_T_array[-1]], [0,100], linestyle="--", color = "green", linewidth = '0.5', label="Change of Domain")
	else:
		plt.plot(N_T_array, A_T_array, color = "blue")
		plt.plot([N_T_array[-1], N_T_array[-1]], [0,100], linestyle="--", color = "green", linewidth = '0.5')

	# Find N_{T,m} and plot
	derivative_values = (A_T_array[1:] - A_T_array[:-1]) / (N_T_array[1:] - N_T_array[:-1])

	# print(derivative_values)
	zeta = 0.04
	bool_array = derivative_values > zeta
	last_true_index = np.where(bool_array)[0][-1]+1

	if first_run_flag:
		first_run_flag = False
		plt.scatter(N_T_array[last_true_index], A_T_array[last_true_index], c="red", marker="o", label="$N_T$")
	else:
		plt.scatter(N_T_array[last_true_index], A_T_array[last_true_index], c="red", marker="o")


	# Calculating AS and plotting
	N_d = 100
	N_T = N_T_array[last_true_index]
	A_T = A_T_array[last_true_index]
	# print(A_T)
	# print(N_T)
	AS = A_T /(N_T-counter*100)

	mean_A_T += A_T_array[-1]
	mean_AS += AS




	# # Write improvements and Adaptability Score 
	# if noise_type == "original":
	# 	text = "Baseline"
	# 	x = N_T_array[0]+15

	# elif noise_type == "contrast":
	# 	text = "Contrast"
	# 	x = N_T_array[0]+20

	# elif noise_type == "defocus_blur":
	# 	text = "Defocus Blur"
	# 	x = N_T_array[0]+15
	
	# elif noise_type == "fog":
	# 	text = "Fog"
	# 	x = N_T_array[0]+35

	# elif noise_type == "zoom_blur":
	# 	text = "Zoom Blur"
	# 	x = N_T_array[0]+20

	# elif noise_type == "original":
	# 	text = "Baseline"
	# 	x = N_T_array[0]+10

	# plt.text(x,35, text)
	# plt.text(N_T_array[0]+5,30, str(round(A_T_array[0], 1))+"% -> "+str(round(A_T_array[-1], 1))+"%")#, size=7)
	# plt.text(N_T_array[0]+15,45, "AS =  "+str(round(AS,2)))

mean_AS = mean_AS/(counter+1)
mean_A_T = mean_A_T/(counter+1)

print("mean A_T = ", mean_A_T)
print("mean AS = ", mean_AS)

plt.legend()
plt.ylim(0,100)
# plt.xlim(0,400)
plt.xlabel("Number of Samples")
plt.ylabel("Top-1 Classification Accuracy (%)")
plt.savefig(results_folder+"/AdaptabilityScore.pdf")


# # ===============================================
# # Adaptability Score 
# # ===============================================

# # Strictly in the order of ["brightness","contrast","defocus_blur","elastic_transform","fog","frost","gaussian_blur","gaussian_noise", "glass_blur", "impulse_noise","jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]
# Source_acuracy = [0.783, 0.270, 0.593, 0.722, 0.510, 0.674,0.5530, 0.6200, 0.5930,0.4160,0.7866,0.6143,0.750,0.7371,0.6386, 0.7186, 0.6691, 0.6204, 0.6252] 

# experiment_id = 64

# file_name = "Results_logs/exp"+str(experiment_id)+".txt"

# df = pd.read_csv(file_name)
# NOISE_TYPES = np.unique(df["noise_type"])

# first_run_flag = True
# last_N_T = 0

# mean_A_T = 0
# mean_AS = 0

# counter = -1
# for noise_type in NOISE_TYPES:
# 	counter+=1
	
# 	df_extracted  = df[df.loc[:,"noise_type"]==noise_type]
# 	A_T_array = df_extracted["A_T"].values
# 	N_T_array = df_extracted["N_T"].values

# 	A_T_array = np.insert(A_T_array[:5], 0, Source_acuracy[counter])
# 	N_T_array = np.insert(N_T_array[:5],0, 0) 
# 	N_T_array[-1] = 100
	
# 	A_T_array = A_T_array *100

# 	N_T_array = N_T_array + last_N_T
# 	last_N_T  = N_T_array[-1]


# 	# Find N_{T,m} and plot
# 	derivative_values = (A_T_array[1:] - A_T_array[:-1]) / (N_T_array[1:] - N_T_array[:-1])

# 	zeta = 0.04
# 	bool_array = derivative_values > zeta
# 	last_true_index = np.where(bool_array)[0][-1]+1

# 	# Calculating AS and plotting
# 	N_T = N_T_array[last_true_index]
# 	A_T = A_T_array[last_true_index]
# 	print(A_T)
# 	print(N_T)
# 	AS = A_T /(N_T-counter*100)

# 	mean_A_T += A_T_array[-1]
# 	mean_AS += AS


# mean_AS = mean_AS/(counter+1)
# mean_A_T = mean_A_T/(counter+1)

# print("mean A_T = ", mean_A_T)
# print("mean AS = ", mean_AS)
