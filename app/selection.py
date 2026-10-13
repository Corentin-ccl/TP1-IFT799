import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Source : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# Source : https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
from sklearn.manifold import TSNE


def acp(df, n=15):
    pca = PCA(n_components=n)
    cp  = pca.fit_transform(df)
    return pd.DataFrame(cp, columns=[f'PC{i+1}' for i in range(n)])


def iter_acp(df, labels, n, pairs):
    results = {}
    results['global'] = acp(df, n)
    results['global']['Class'] = labels['Class'].to_numpy()

    for pair in pairs:
        mask = labels['Class'].isin(pair)
        key = f'{pair[0]}-{pair[1]}'
        results[key] = acp(df[mask], n)
        results[key]['Class'] = labels.loc[mask, 'Class'].to_numpy()

    return results


def best_2_cp_acp(df_acp, labels):
    cp_pairs = list(itertools.combinations(df_acp.columns, 2))

    best_pair = None
    best_overlap = float('inf')
    for pair in cp_pairs:
        cp_data = df_acp[list(pair)].to_numpy()
        overlap = ...
        if overlap < best_overlap:
            best_overlap = overlap
            best_pair = pair

    return best_pair


def show_acp(key, df_acp):
    best_pair = best_2_cp_acp(df_acp.drop(columns='Class'), df_acp['Class'])

    plt.figure()
    for tumor in df_acp['Class'].unique():
        points = df_acp[df_acp['Class'] == tumor]
        plt.scatter(points[best_pair[0]], points[best_pair[1]], label=tumor)
    plt.title(f'ACP: {key}')
    plt.xlabel(f'{best_pair[0]}')
    plt.ylabel(f'{best_pair[1]}')
    plt.legend()
    plt.show()


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