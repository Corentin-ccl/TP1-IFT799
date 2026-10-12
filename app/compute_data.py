import os
import pandas as pd
import numpy as np


def distance_euclidienne(point1, point2):
    """
    Calcule la distance euclidienne entre deux points.

    Parameters:
    point1 (np.ndarray): Le premier point.
    point2 (np.ndarray): Le deuxième point.

    Returns:
    float: La distance euclidienne entre les deux points.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

def distance_mahalanobis(point1, point2, covariance_matrix):
    """
    Calcule la distance de Mahalanobis entre deux points.

    Parameters:
    point1 (np.ndarray): Le premier point.
    point2 (np.ndarray): Le deuxième point.
    covariance_matrix (np.ndarray): La matrice de covariance.

    Returns:
    float: La distance de Mahalanobis entre les deux points.
    """
    diff = point1 - point2
    inv_cov_matrix = np.linalg.inv(covariance_matrix)
    return np.sqrt(np.dot(np.dot(diff.T, inv_cov_matrix), diff))

def distance_cosine(point1, point2):
    """
    Calcule la distance cosinus entre deux points.

    Parameters:
    point1 (np.ndarray): Le premier point.
    point2 (np.ndarray): Le deuxième point.

    Returns:
    float: La distance cosinus entre les deux points.
    """
    dot_product = np.dot(point1, point2)
    norm_point1 = np.linalg.norm(point1)
    norm_point2 = np.linalg.norm(point2)
    
    if norm_point1 == 0 or norm_point2 == 0:
        return 0.0  # Évite la division par zéro
    
    cosine_similarity = dot_product / (norm_point1 * norm_point2)
    return 1 - cosine_similarity  # Distance cosinus

def distance_intraClasse(data, distance_function):
    numeric_data = data.select_dtypes(include='number')
    
    if len(numeric_data) <= 1:
        return 0.0

    data_values = numeric_data.values
    func_name = distance_function.__name__
    
    if func_name == 'distance_cosine':
        data_values = data_values / np.linalg.norm(data_values, axis=1, keepdims=True)
        mean_vector = np.mean(data_values, axis=0)
    elif func_name == 'distance_mahalanobis':
        mean_vector = np.mean(data_values, axis=0)
        covariance_matrix = np.cov(data_values.T)
    else:
        mean_vector = np.mean(data_values, axis=0)

    intra_class_distances = []
    
    for point in data_values:
        if func_name == 'distance_mahalanobis':
            dist = distance_function(point, mean_vector, covariance_matrix)
        else:
            dist = distance_function(point, mean_vector)
            
        intra_class_distances.append(dist)
    
    return np.max(intra_class_distances)

def distance_interClasse(data_class1, data_class2, distance_function):
    import numpy as np
import pandas as pd

def distance_interClasse(data_class1, data_class2, distance_function):
    """
    Calcule la distance inter-classe entre deux classes de données.
    """
    val_c1 = data_class1.select_dtypes(include='number').values
    val_c2 = data_class2.select_dtypes(include='number').values
    func_name = distance_function.__name__
    
    if func_name == 'distance_cosine':
        val_c1 = val_c1 / np.linalg.norm(val_c1, axis=1, keepdims=True)
        val_c2 = val_c2 / np.linalg.norm(val_c2, axis=1, keepdims=True)
        
    center1 = np.mean(val_c1, axis=0)
    center2 = np.mean(val_c2, axis=0)
    
    cov1, cov2 = None, None
    if func_name == 'distance_mahalanobis':
        cov1 = np.cov(val_c1.T)
        cov2 = np.cov(val_c2.T)

    dist_c1_c2_list = []
    for x in val_c1:
        if func_name == 'distance_mahalanobis':
            d = distance_function(x, center2, cov2)
        else:
            d = distance_function(x, center2)
        dist_c1_c2_list.append(d)
        
    dist_c1_c2 = np.min(dist_c1_c2_list)

    dist_c2_c1_list = []
    for y in val_c2:
        if func_name == 'distance_mahalanobis':
            d = distance_function(y, center1, cov1)
        else:
            d = distance_function(y, center1)
        dist_c2_c1_list.append(d)
        
    dist_c2_c1 = np.min(dist_c2_c1_list)

    return min(dist_c1_c2, dist_c2_c1)


def overlap_coefficient(data, labels):
    """
    Calcule le coefficient de chevauchement entre les classes dans les données.

    Parameters:
    data (pd.DataFrame): Les données d'entrée.
    labels (pd.Series): Les étiquettes de classe correspondantes.

    Returns:
    dict: Un dictionnaire contenant les coefficients de chevauchement pour chaque paire de classes.
    """
    unique_labels = np.unique(labels)
    overlap_coefficients = {}
    
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            class1_data = data[labels == unique_labels[i]]
            class2_data = data[labels == unique_labels[j]]
            
            # Calcul du coefficient de chevauchement
            overlap = np.sum(np.minimum(class1_data, class2_data)) / np.sum(np.maximum(class1_data, class2_data))
            overlap_coefficients[(unique_labels[i], unique_labels[j])] = overlap
    
    return overlap_coefficients


if __name__ == "__main__":
    data_path="C:\\travail\\Sherbrooke\\ScienceDesDonnees\\tp\\TP1\\TP1-IFT799\\data\\prepared_data_BRCA.csv"
    list_data_path= os.listdir("C:\\travail\\Sherbrooke\\ScienceDesDonnees\\tp\\TP1\\TP1-IFT799\\data")
    list_distances = ['distance_euclidienne', 'distance_mahalanobis', 'distance_cosine']
    for file_name in list_data_path:
        if file_name.startswith("prepared_data_") and file_name.endswith(".csv"):
            data_path = os.path.join("C:\\travail\\Sherbrooke\\ScienceDesDonnees\\tp\\TP1\\TP1-IFT799\\data", file_name)
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                print(f"Calcul des distances intra-classe pour {file_name}:")
                print(data.head())
                print(data.shape)
                result = distance_intraClasse(data, distance_function=distance_euclidienne)
                print(result)
                print("----------------------------------------------------------------------------------------------------------")
    data_path1="C:\\travail\\Sherbrooke\\ScienceDesDonnees\\tp\\TP1\\TP1-IFT799\\data\\prepared_data_BRCA.csv"
    data_path2="C:\\travail\\Sherbrooke\\ScienceDesDonnees\\tp\\TP1\\TP1-IFT799\\data\\prepared_data_PRAD.csv"
    data1 = pd.read_csv(data_path1)
    data2 = pd.read_csv(data_path2)
    print(f"Calcul des distances inter-classe pour BRCA et PRAD:")
    result = distance_interClasse(data1, data2, distance_function=distance_euclidienne)
    print(result)
