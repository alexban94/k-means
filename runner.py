import numpy as np
import matplotlib as plt
import tensorflow as tf
import pandas as pd
import K_Means as km

df = pd.read_csv('Wholesale customers data.csv')
df = df.drop(0)
print(df)

mu_k, r_nk = km.k_means(df, 3)