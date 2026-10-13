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
    shape_cov = covariance_matrix.shape
    if len(shape_cov) != 2 or shape_cov[0] != shape_cov[1]:
        inv_cov_matrix = np.linalg.pinv(covariance_matrix)
        
    else:
        try:
            inv_cov_matrix = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            inv_cov_matrix = np.linalg.pinv(covariance_matrix)

    dist = np.sqrt(np.dot(np.dot(diff.T, inv_cov_matrix), diff))
    dist_carre = max(0.0, dist)

    return np.sqrt(dist_carre)


def distance_intraClasse(data, distance_function):
    """
    Calcule la distance intra-classe pour un ensemble de données donné en utilisant une fonction de distance spécifiée.

    Parameters:
    data (pd.DataFrame): Les données de la classe.
    distance_function (function): La fonction de distance à utiliser.

    Returns:
    float: La distance intra-classe maximale.
    """
    func_name = distance_function.__name__
    numeric_data = data.select_dtypes(include='number')

    if func_name == 'distance_mahalanobis':
        n_samples = numeric_data.shape[0]
        n_cols = max(1, n_samples - 10)
        if n_cols < numeric_data.shape[1]:
            np.random.seed(42)
            colonnes_aleatoires = np.random.choice(numeric_data.columns, size=n_cols, replace=False)
            numeric_data = numeric_data[colonnes_aleatoires]

    if len(numeric_data) <= 1:
        return 0.0
    data_values = numeric_data.values
    
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

    Parameters:
    data_class1 (pd.DataFrame): Les données de la première classe.
    data_class2 (pd.DataFrame): Les données de la deuxième classe.
    distance_function (function): La fonction de distance à utiliser.

    Returns:
    float: La distance inter-classe entre les deux classes.
    """
    dfclass1 = data_class1.select_dtypes(include='number')
    dfclass2 = data_class2.select_dtypes(include='number')
    func_name = distance_function.__name__

    if func_name == 'distance_mahalanobis':
        n_samples = min(dfclass1.shape[0], dfclass2.shape[0])
        n_cols = max(1, n_samples - 1)
        if n_cols < dfclass1.shape[1]:
            np.random.seed(42)
            colonnes_aleatoires = np.random.choice(dfclass1.columns, size=n_cols, replace=False)
            dfclass1 = dfclass1[colonnes_aleatoires]
            dfclass2 = dfclass2[colonnes_aleatoires]

    val_c1 = dfclass1.values
    val_c2 = dfclass2.values
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
           

def tab_result( output_file="resultats_distance.csv"):
    """
    Crée un tableau de résultats contenant les distances intra-classe, inter-classe et l'overlap pour différentes classes et méthodes de distance.
    
    Parameters:
    output_file (str): Le chemin du fichier de sortie pour sauvegarder le tableau de résultats
    
    Returns:
    pd.DataFrame: Le tableau de résultats contenant les distances et l'overlap pour chaque combinaison de classes et de méthodes de distance."""

    results = []
    
    distance_functions = [distance_euclidienne, distance_mahalanobis]
    class_list = ["BRCA", "PRAD", "KIRC", "LUAD", "COAD"]

    class_data_dict = {}
    for class_name in class_list:
        file_path = f"basedir\\data\\prepared_data_{class_name}.csv"
        class_data_dict[class_name] = pd.read_csv(file_path)
    for dist_func in distance_functions:
        func_name = dist_func.__name__

        for i in range(len(class_list)):
            for j in range(i + 1, len(class_list)):
                c1_name = class_list[i]
                c2_name = class_list[j]
                
                data_c1 = class_data_dict[c1_name]
                data_c2 = class_data_dict[c2_name]
                
                dist_intra_c1 = distance_intraClasse(data_c1, dist_func)
                dist_intra_c2 = distance_intraClasse(data_c2, dist_func)
                dist_inter = distance_interClasse(data_c1, data_c2, dist_func)
                overlap = overlap_coefficient(data_c1, data_c2, dist_func)
                
                results.append({
                    'Methode_Distance': func_name,
                    'Classe_1': c1_name,
                    'Classe_2': c2_name,
                    'Intra_Classe_1': dist_intra_c1,
                    'Intra_Classe_2': dist_intra_c2,
                    'Inter_Classe': dist_inter,
                    'Overlap': overlap
                })
    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        df_results.to_csv(output_file, index=False)
        print(f"Tableau de {len(df_results)} lignes sauvegardé avec succès dans : {output_file}")
        
    return df_results


if __name__ == "__main__":
    tab_result(output_file="resultats_distance.csv")
