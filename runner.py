import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from k_means import k_means
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Load data and scale it
df = pd.read_csv('customer_kaggle.csv', index_col=0)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna()
#df = sns.load_dataset('iris')
#df = df.drop(labels = 'species', axis = 1)
print(df)
scaler = MinMaxScaler()
np_data = scaler.fit_transform(df.dropna())

# Perform PCA to keep only 2 principal components for scatter plot visualization
reduced_data = PCA(n_components=2, whiten=True).fit_transform(np_data)


# Conduct K-Means on the reduced data
max_iter = 100
mu_k, r_nk = k_means(reduced_data[0:100,:], 3, max_iter)
print(mu_k)
