import pandas as pd
import numpy as np

# Source : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

def acp(df, n):
    pca = PCA(n_components=n)
    cp  = pca.fit_transform(df)
    return pd.DataFrame(cp, columns=[f'PC{i+1}' for i in range(n)])