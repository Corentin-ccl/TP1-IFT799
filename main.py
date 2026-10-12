import pandas as pd
import numpy as np
import itertools

from selection import acp, iter_tsne, show_tsne

N_COLS = 200

data   = pd.read_csv('data.csv')
labels = pd.read_csv('labels.csv')

tumors = labels['Class'].unique()
pairs  = list(itertools.combinations(tumors, 2))

def reduce_dim_random(df, n):
    keep = np.random.choice(df.columns, n, replace=False)
    return df[keep]

reduced = reduce_dim_random(data, N_COLS)

# ACP
# acp_data = acp(reduced, 15)

# TODO : Overlap results ACP

# t-SNE
tsne_data = iter_tsne(reduced, labels, 2, pairs)
for key, df in tsne_data.items():
    show_tsne(key, df)