import pandas as pd
import numpy as np

from selection import acp

data   = pd.read_csv('data.csv')
labels = pd.read_csv('labels.csv')

print(data.shape)
print(labels.shape)

def reduce_dim_random(df, n):
    keep = np.random.choice(df.columns, n, replace=False)
    return df[keep]

data2000 = reduce_dim_random(data, 2000)
print(data2000.shape)

acp_data = acp(data2000, 15)
print(acp_data.head())
