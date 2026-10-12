import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Source : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# Source : https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
from sklearn.manifold import TSNE


def acp(df, n=15):
    pca = PCA(n_components=n)
    cp  = pca.fit_transform(df)
    return pd.DataFrame(cp, columns=[f'PC{i+1}' for i in range(n)])


def tsne(df, n=2):
    t_sne = TSNE(n_components=n)
    cp    = t_sne.fit_transform(df)
    return pd.DataFrame(cp, columns=[f'tSNE{i+1}' for i in range(n)])

def iter_tsne(df, labels, n, pairs):
    results = {}
    results['global'] = tsne(df, n)
    results['global']['Class'] = labels['Class'].to_numpy()

    for pair in pairs:
        mask = labels['Class'].isin(pair)
        key = f'{pair[0]}-{pair[1]}'
        results[key] = tsne(df[mask], n)
        results[key]['Class'] = labels.loc[mask, 'Class'].to_numpy()

    return results

def show_tsne(key, df_tsne):
    plt.figure()
    for tumor in df_tsne['Class'].unique():
        points = df_tsne[df_tsne['Class'] == tumor]
        plt.scatter(points.iloc[:, 0], points.iloc[:, 1], label=tumor)
    plt.title(f't-SNE: {key}')
    plt.xlabel('tSNE1')
    plt.ylabel('tSNE2')
    plt.legend()
    plt.show()