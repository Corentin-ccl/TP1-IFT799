import pandas as pd
import itertools

from prepare_data import reduce_dim_random
from selection import find_best_pair, find_best_pair_global, show, acp, find_best_n_acp, tsne, iter_method
from compute_data import overlap_coefficient, distance_euclidienne

N_COLS = 200
DATA_FOLDER = '../data/'

data   = pd.read_csv(DATA_FOLDER + 'data.csv')
labels = pd.read_csv(DATA_FOLDER + 'labels.csv')

tumors = labels['Class'].unique() # = ['BRCA', 'KIRC', 'COQD', 'LUAD', 'PRAD']
pairs = list(itertools.combinations(tumors, 2))

df = pd.merge(data, labels, on='Unnamed: 0')
df = reduce_dim_random(df, N_COLS)


# Paires originales
## Global
global_best_pair, global_best_overlap = find_best_pair_global(df)
print(f'Global best pair: {global_best_pair}, Overlap: {global_best_overlap}')
show(df[[global_best_pair[0], global_best_pair[1], 'Class']], 'Global best pair', global_best_pair[0], global_best_pair[1])

## Paires de classes
for pair in pairs:
    best_pair, best_overlap = find_best_pair(df, pair[0], pair[1])
    print(f'Best pair for {pair[0]}-{pair[1]}: {best_pair}, Overlap: {best_overlap}')
    show(df[df['Class'].isin(pair)][[best_pair[0], best_pair[1], 'Class']], f'Best pair for {pair[0]}-{pair[1]}', best_pair[0], best_pair[1])


# ACP
best_n_acp = find_best_n_acp(df, treshold=0.95, show=True)

acp_data = iter_method(df, pairs, method=acp, n=best_n_acp)
for key, df_acp in acp_data.items():
    show(df_acp[['PC1', 'PC2', 'Class']], f'ACP: {key}', 'PC1', 'PC2')

    ## Global
    if key == 'global':
        best_pair, best_overlap = find_best_pair_global(df_acp)
        print(f'Best pair for global: {best_pair}, Overlap: {best_overlap}')
        show(df_acp[[best_pair[0], best_pair[1], 'Class']], f'ACP : Global best pair', best_pair[0], best_pair[1])
        continue

    ## Paires de classes
    class1 = key.split('-')[0]
    class2 = key.split('-')[1]
    best_pair, best_overlap = find_best_pair(df_acp, class1, class2)
    print(f'Best pair for {class1}-{class2}: {best_pair}, Overlap: {best_overlap}')
    show(df_acp[[best_pair[0], best_pair[1], 'Class']], f'ACP : Best pair for {class1}-{class2}', best_pair[0], best_pair[1])


# t-SNE
tsne_data = iter_method(df, pairs, method=tsne, n=2)
for key, df_tsne in tsne_data.items():
    overlap = overlap_coefficient(df_tsne[df_tsne['Class'] == df_tsne['Class'].unique()[0]][['tSNE1', 'tSNE2']], df_tsne[df_tsne['Class'] == df_tsne['Class'].unique()[1]][['tSNE1', 'tSNE2']], distance_euclidienne)
    print(f't-SNE: {key} - Overlap: {overlap}')
    show(df_tsne, f't-SNE: {key}', 'tSNE1', 'tSNE2')