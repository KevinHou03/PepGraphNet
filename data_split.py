import ast
import numpy as np
import pandas as pd
from rdkit import Chem

# load data
data = pd.read_csv('/Users/kevinhou/Documents/CY Lab/Data/CycPeptMPDB_Peptide_Cleaned.csv')
output_dir = '/Users/kevinhou/Documents/CY Lab/Data/'

# split into circle/lariat and with each assay and all circle/lariat regardless of its assay
datasets = {
    'lariat_caco2': data[(data['Caco2'].notna()) & (data['Molecule_Shape'] == 'Lariat')],
    'circle_caco2': data[(data['Caco2'].notna()) & (data['Molecule_Shape'] == 'Circle')],
    'lariat_pampa': data[(data['PAMPA'].notna()) & (data['Molecule_Shape'] == 'Lariat')],
    'circle_pampa': data[(data['PAMPA'].notna()) & (data['Molecule_Shape'] == 'Circle')],
    'lariat_rrck': data[(data['RRCK'].notna()) & (data['Molecule_Shape'] == 'Lariat')],
    'circle_rrck': data[(data['RRCK'].notna()) & (data['Molecule_Shape'] == 'Circle')],
    'lariat_mdck': data[(data['MDCK'].notna()) & (data['Molecule_Shape'] == 'Lariat')],
    'circle_mdck': data[(data['MDCK'].notna()) & (data['Molecule_Shape'] == 'Circle')],
    'lariat_all': data[data['Molecule_Shape'] == 'Lariat'],
    'circle_all': data[data['Molecule_Shape'] == 'Circle']
}

# 按照train/val/test划分为0.8/0.8/0.1
def random_split(dataset, dataset_name, package_path):
    path_prefix = f"{package_path}{dataset_name}"
    rand_data = dataset.sample(frac=1, random_state=13).reset_index(drop=True)
    length = dataset.shape[0]

    train_end = int(0.8 * length)
    val_end = int(0.9 * length)

    train = rand_data.iloc[:train_end]
    val = rand_data.iloc[train_end:val_end]
    test = rand_data.iloc[val_end:]

    train.to_csv(f"{path_prefix}_train.csv", index=False)
    val.to_csv(f"{path_prefix}_val.csv", index=False)
    test.to_csv(f"{path_prefix}_test.csv", index=False)

# # test
# for name, df in datasets.items():
#     random_split(df, name, output_dir)









