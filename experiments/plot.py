import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd


file_name = "Results_logs/exp2.txt"


df = pd.read_csv(file_name)
NOISE_TYPES = np.unique(df["noise_type"])

for noise_type in NOISE_TYPES:
	df_extracted  = df[df.loc[:,"noise_type"]==noise_type]
	N_T = df_extracted["N_T"].values
	A_T = df_extracted["A_T"].values

	# print(N_T)
	plt.plot(N_T, A_T, label=noise_type)
	plt.legend()
plt.xlabel("N_T")
plt.ylabel("A_T")
plt.savefig("exp2.pdf")
	# print(df[df.loc[:,"noise_type"]==noise_type])
	# input("enter")