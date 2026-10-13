import os
import pandas as pd
import numpy as np

def segmentation_csv(data, lignes_par_fichier=5):
    """
    Segmentation of the CSV file into smaller files based on the 'segment' column.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data to be segmented.

    Returns:
    None
    """
    output_dir = "segmented_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    

    for i in range(0, len(data), lignes_par_fichier):
        chunk = data.iloc[i:i + lignes_par_fichier]
        numero_fichier = (i // lignes_par_fichier) + 1
        output_file = os.path.join(output_dir, f"partie_{numero_fichier}.csv")
        chunk.to_csv(output_file, index=False)

def join_dfs_on_column(df1, df2, column_name):
    """
    Joins two DataFrames on a specified column.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    column_name (str): The name of the column to join on.

    Returns:
    pd.DataFrame: The joined DataFrame.
    """
    if column_name not in df1.columns or column_name not in df2.columns:
        raise ValueError(f"Both DataFrames must contain the column '{column_name}'.")

    joined_df = pd.merge(df1, df2, on=column_name)

    return joined_df

def get_data_from_class(data, labels, target_class):
    """
    Récupère les données correspondant à une classe spécifique.

    Parameters:
    data (pd.DataFrame): Les données d'entrée.
    labels (pd.Series): Les étiquettes de classe correspondantes.
    target_class: La classe cible pour laquelle récupérer les données.

    Returns:
    pd.DataFrame: Un DataFrame contenant uniquement les données de la classe cible.
    """
    if target_class not in labels.values:
        raise ValueError(f"La classe '{target_class}' n'existe pas dans les étiquettes.")

    class_data = data[labels == target_class]
    
    return class_data

def reduce_dim_random(df, n):
    out = pd.DataFrame()
    out['Unnamed: 0'] = df['Unnamed: 0']
    out['Class']      = df['Class']
    keep = np.random.choice(df.columns, n, replace=False)
    out = pd.concat([out, df[keep]], axis=1)
    return out


if __name__ == "__main__":
    data_path1="basedir\\data\\data.csv"
    data_path2="basedir\\data\\labels.csv"

    if os.path.exists(data_path1) and os.path.isfile(data_path2):
        class_list=['PRAD', 'BRCA', 'KIRC', 'COAD', 'LUAD']
        data1 = pd.read_csv(data_path1)
        data2 = pd.read_csv(data_path2)
        data = join_dfs_on_column(data1, data2, 'Unnamed: 0')
        print(data.head())
        print(data.shape)
        for class_name in class_list:
            prepared_data = get_data_from_class(data, data['Class'], class_name)
            print(f"Data for class {class_name}:")
            print(prepared_data.head())
            print(prepared_data.shape)
            prepared_data.to_csv(f'basedir\\data\\prepared_data_{class_name}.csv', index=False)
        #segmentation_csv(data1)