import pandas as pd
import numpy as np
import itertools

from selection import acp, iter_acp, iter_tsne, show_acp, show_tsne

N_COLS = 200

data   = pd.read_csv('../data/data.csv')
labels = pd.read_csv('../data/labels.csv')

data   = data.drop(data.columns[0], axis=1)

tumors = labels['Class'].unique()
pairs  = list(itertools.combinations(tumors, 2))

def reduce_dim_random(df, n):
    keep = np.random.choice(df.columns, n, replace=False)
    return df[keep]

reduced = reduce_dim_random(data, N_COLS)

# ACP
acp_data = iter_acp(reduced, labels, 15, pairs)
for key, df in acp_data.items():
    show_acp(key, df)

# t-SNE
tsne_data = iter_tsne(reduced, labels, 2, pairs)
for key, df in tsne_data.items():
    show_tsne(key, df)