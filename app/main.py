import pandas as pd
import numpy as np
import itertools

from prepare_data import reduce_dim_random
from selection import acp, find_best_pair_global, tsne, iter_method, show_tsne, find_best_pair

N_COLS = 200
DATA_FOLDER = '../data/'

data   = pd.read_csv(DATA_FOLDER + 'data.csv')
labels = pd.read_csv(DATA_FOLDER + 'labels.csv')

tumors = labels['Class'].unique() # = ['BRCA', 'KIRC', 'COQD', 'LUAD', 'PRAD']

df = pd.merge(data, labels, on='Unnamed: 0')
df = reduce_dim_random(df, N_COLS)


# Paires originales
pairs = list(itertools.combinations(tumors, 2))
for pair in pairs:
    best_pair, best_overlap = find_best_pair(df, pair[0], pair[1])
    print(f'Best pair for {pair[0]}-{pair[1]}: {best_pair}, Overlap: {best_overlap}')

global_best_pair, global_best_overlap = find_best_pair_global(df)
print(f'Global best pair: {global_best_pair}, Overlap: {global_best_overlap}')


# ACP
# acp_data = iter_method(df, pairs, method=acp, n=15)
# for key, df_acp in acp_data.items():
#     show_acp(key, df_acp)

# t-SNE
tsne_data = iter_method(df, pairs, method=tsne, n=2)
for key, df_tsne in tsne_data.items():
    show_tsne(key, df_tsne)