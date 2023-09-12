import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

experiment_id = 66

file_name = "Results_logs/exp"+str(experiment_id)+".txt"


df = pd.read_csv(file_name)
NOISE_TYPES = np.unique(df["noise_type"])

for noise_type in NOISE_TYPES:
	df_extracted  = df[df.loc[:,"noise_type"]==noise_type]
	N_T_array = df_extracted["N_T"].values
	A_T_array = df_extracted["A_T"].values

	# print(N_T)
	plt.plot(N_T_array, A_T_array, label=noise_type)
	plt.legend()
plt.ylim(0.1,0.8)
plt.xlabel("N_T")
plt.ylabel("A_T")
plt.savefig("exp"+str(experiment_id)+".pdf")
	# print(df[df.loc[:,"noise_type"]==noise_type])
	# input("enter")


# ============================
# Calculate Adaptability Score
# ============================
N_d = 152

M  = len(NOISE_TYPES)
AS = 0

for noise_type in NOISE_TYPES:
	
	df_extracted  = df[df.loc[:,"noise_type"]==noise_type]

	N_T_array = df_extracted["N_T"].values
	A_T_array = df_extracted["A_T"].values

	print(N_T_array[N_T_array== N_d])
	print(A_T_array[N_T_array== N_d])

	N_T = N_T_array[N_T_array== N_d]
	A_T = A_T_array[N_T_array== N_d]
	# AS  += (1/M) *  A_T[0]*N_d/N_T
	AS  += (A_T[0]*N_d/N_T)/M
	print(AS)
	print("===")


