import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from compute_data import overlap_coefficient, distance_euclidienne

# Source : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# Source : https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
from sklearn.manifold import TSNE


def find_best_pair(df, class1, class2):
    columns = df.columns.drop(['Unnamed: 0', 'Class'], errors='ignore')
    class1_data = df[df['Class'] == class1][columns]
    class2_data = df[df['Class'] == class2][columns]

    best_col_1     = None
    best_overlap_1 = float('inf')
    for col in columns:
        overlap = overlap_coefficient(class1_data[[col]], class2_data[[col]], distance_euclidienne)
        if overlap < best_overlap_1:
            best_overlap_1 = overlap
            best_col_1 = col

    best_col_2     = None
    best_overlap_2 = float('inf')
    for col in columns:
        if col == best_col_1:
            continue
        overlap = overlap_coefficient(class1_data[[best_col_1, col]], class2_data[[best_col_1, col]], distance_euclidienne)
        if overlap < best_overlap_2:
            best_overlap_2 = overlap
            best_col_2 = col

    return (best_col_1, best_col_2), best_overlap_2


# Pareil que au dessus mais on compare les moyennes pour toutes les paires de classes
def find_best_pair_global(df):
    columns = df.columns.drop(['Unnamed: 0', 'Class'], errors='ignore')

    class_pairs = list(itertools.combinations(df['Class'].unique(), 2))

    best_col_1     = None
    best_overlap_1 = float('inf')
    for col in columns:
        overlap_sum = 0
        for class1, class2 in class_pairs:
            class1_data = df[df['Class'] == class1][[col]]
            class2_data = df[df['Class'] == class2][[col]]

            overlap = overlap_coefficient(class1_data, class2_data, distance_euclidienne)

            overlap_sum += overlap
        global_overlap = overlap_sum / len(class_pairs)

        if global_overlap < best_overlap_1:
            best_overlap_1 = global_overlap
            best_col_1 = col

    best_col_2     = None
    best_overlap_2 = float('inf')
    for col in columns:
        if col == best_col_1:
            continue
        overlap_sum = 0
        for class1, class2 in class_pairs:
            class1_data = df[df['Class'] == class1][[best_col_1, col]]
            class2_data = df[df['Class'] == class2][[best_col_1, col]]

            overlap = overlap_coefficient(class1_data, class2_data, distance_euclidienne)

            overlap_sum += overlap
        global_overlap = overlap_sum / len(class_pairs)

        if global_overlap < best_overlap_2:
            best_overlap_2 = global_overlap
            best_col_2 = col

    return (best_col_1, best_col_2), best_overlap_2
    

def find_best_n_acp(df, treshold=0.95, show=False):
    df = df.drop(columns=['Unnamed: 0', 'Class'], errors='ignore')
    pca = PCA(n_components=df.shape[1])
    pca.fit(df)
    explained_variance = pca.explained_variance_ratio_

    cumulative_variance = np.cumsum(explained_variance)
    n_components = np.argmax(cumulative_variance >= treshold) + 1

    if show:
        plt.figure()
        plt.plot(cumulative_variance, marker='o')
        plt.axhline(y=treshold, color='r', linestyle='--')
        plt.axvline(x=n_components - 1, color='g', linestyle='--')
        plt.title('Variance expliquée cumulée par ACP')
        plt.xlabel('Nombre de composantes principales')
        plt.ylabel('Variance expliquée cumulée')
        plt.grid()
        plt.show()

    return n_components


def acp(df, n=15):
    df = df.drop(columns=['Unnamed: 0', 'Class'], errors='ignore')
    pca = PCA(n_components=n)
    cp  = pca.fit_transform(df)
    return pd.DataFrame(cp, columns=[f'PC{i+1}' for i in range(n)])


def tsne(df, n=2):
    df = df.drop(columns=['Unnamed: 0', 'Class'], errors='ignore')
    t_sne = TSNE(n_components=n)
    cp    = t_sne.fit_transform(df)
    return pd.DataFrame(cp, columns=[f'tSNE{i+1}' for i in range(n)])


def iter_method(df, pairs, method=acp, n=15):
    results = {}
    results['global'] = method(df, n)
    results['global']['Class'] = df['Class'].to_numpy()

    for pair in pairs:
        mask = df['Class'].isin(pair)
        key = f'{pair[0]}-{pair[1]}'
        results[key] = method(df[mask], n)
        results[key]['Class'] = df.loc[mask, 'Class'].to_numpy()

    return results


def show(df, title, x_label, y_label):
    plt.figure()
    for tumor in df['Class'].unique():
        points = df[df['Class'] == tumor]
        plt.scatter(points.iloc[:, 0], points.iloc[:, 1], label=tumor)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
    