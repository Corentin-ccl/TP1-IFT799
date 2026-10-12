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
    
    if func_name == 'distance_mahalanobis':
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


def overlap_coefficient(data_class1, data_class2, distance_function):
    """
    Calcule l'indicateur de superposition (Overlap) entre les classes dans les données.
    
    Parameters:
    data_class1 (pd.DataFrame): Les données de la première classe.
    data_class2 (pd.DataFrame): Les données de la deuxième classe.
    distance_function (function): La fonction de distance à utiliser.

    Returns:
    float: Le coefficient de chevauchement entre les deux classes.
    """

    dist_intra_c1 = distance_intraClasse(data_class1, distance_function)
    dist_intra_c2 = distance_intraClasse(data_class2, distance_function)

    dist_inter_c1_c2 = distance_interClasse(data_class1, data_class2, distance_function)

    if dist_inter_c1_c2 == 0:
        overlap = float('inf')
    else:
        overlap = (dist_intra_c1 + dist_intra_c2) / (2 * dist_inter_c1_c2)

    return overlap
           

def tab_result(file_list, output_file="resultats_distance.csv"):
    """
    Crée un tableau de résultats contenant les distances intra-classe et inter-classe pour chaque paire de classes.

    Parameters:
    file_list (list): Une liste de chemins vers les fichiers de données.
    output_file (str): Le nom du fichier de sortie.

    Returns:
    pd.DataFrame: Un DataFrame contenant les résultats.
    """
    results = []
    distance_functions = ["distance_euclidienne", "distance_mahalanobis", "distance_cosine"]
    class_list = ['PRAD', 'BRCA', 'KIRC', 'COAD', 'LUAD']
    
    for file_path in file_list:
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)

            for distance_function in distance_functions:
                intra_distances = {}
                inter_distances = {}
                overlap_coeffs = {}

                for className in class_list:
                    class_data = data[data['Class'] == className]
                    intra_distance = distance_intraClasse(class_data, distance_function)
                    intra_distances[className] = intra_distance

                for i in range(len(class_list)):
                    for j in range(i + 1, len(class_list)):
                        class1_data = data[data['Class'] == class_list[i]]
                        class2_data = data[data['Class'] == class_list[j]]
                        inter_distance = distance_interClasse(class1_data, class2_data, distance_function)
                        inter_distances[(class_list[i], class_list[j])] = inter_distance

                results.append({
                    'File': os.path.basename(file_path),
                    'DistanceFunction': distance_function.__name__,
                    'IntraDistances': intra_distances,
                    'InterDistances': inter_distances
                })

    for i in range(len(class_list)):
        class1_data = data[data['Class'] == class_list[i]]
        intra_distance = distance_intraClasse(class1_data, distance_function)

        for j in range(i + 1, len(class_list)):
            class2_data = data[data['Class'] == class_list[j]]
            inter_distance = distance_interClasse(class1_data, class2_data, distance_function)

            results.append({
                'Class1': class_list[i],
                'Class2': class_list[j],
                'IntraDistance_Class1': intra_distance,
                'IntraDistance_Class2': distance_intraClasse(class2_data, distance_function),
                'InterDistance': inter_distance
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    data_path="C:\\travail\\Sherbrooke\\ScienceDesDonnees\\tp\\TP1\\TP1-IFT799\\data\\prepared_data_BRCA.csv"
    list_data_path= os.listdir("C:\\travail\\Sherbrooke\\ScienceDesDonnees\\tp\\TP1\\TP1-IFT799\\data")
    list_distances = ['distance_euclidienne', 'distance_mahalanobis']
    # for file_name in list_data_path:
    #     if file_name.startswith("prepared_data_") and file_name.endswith(".csv"):
    #         data_path = os.path.join("C:\\travail\\Sherbrooke\\ScienceDesDonnees\\tp\\TP1\\TP1-IFT799\\data", file_name)
    #         if os.path.exists(data_path):
    #             data = pd.read_csv(data_path)
    #             print(f"Calcul des distances intra-classe pour {file_name}:")
    #             print(data.head())
    #             print(data.shape)
    #             result = distance_intraClasse(data, distance_function=distance_euclidienne)
    #             print(result)
    #             print("----------------------------------------------------------------------------------------------------------")
    data_path1="C:\\travail\\Sherbrooke\\ScienceDesDonnees\\tp\\TP1\\TP1-IFT799\\data\\prepared_data_BRCA.csv"
    data_path2="C:\\travail\\Sherbrooke\\ScienceDesDonnees\\tp\\TP1\\TP1-IFT799\\data\\prepared_data_PRAD.csv"
    data1 = pd.read_csv(data_path1)
    data2 = pd.read_csv(data_path2)
    print(f"Calcul des distances inter-classe pour BRCA et PRAD:")
    result_intra = distance_intraClasse(data1, distance_function=distance_mahalanobis)
    print(f"Distance intra-classe pour BRCA: {result_intra}")
    # result = distance_interClasse(data1, data2, distance_function=distance_mahalanobis)
    # print(result)
    # result_overlap = overlap_coefficient(data1, data2, distance_function=distance_mahalanobis)
    # print(f"Calcul de l'indicateur de superposition (Overlap) pour BRCA et PRAD:")
    # print(result_overlap)
