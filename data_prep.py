from cgitb import reset

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

# load
data = pd.read_csv('/Users/kevinhou/Documents/CY Lab/Data/CycPeptMPDB_Peptide_All (1).csv', low_memory=False) # this address mixed datatype
print(data.shape) # 8466, 247

# drop NA Permeability
data.dropna(subset=['Permeability'], inplace=True) # Same_Peptides_Permeability or Permeability?
print(data.shape) # 912, 247

# drop NA Assay
# data.dropna(subset=['Same_Peptides_Assay'], inplace=True) # Same_Peptides_Permeability or Permeability?
# print(data.shape) # 912, 247

# drop Permeability < 10-8 or > 10-4 cm/s, filter log_10 permeability values between -8 and -4
data = data[(data['Permeability'] >= -8) & (data['Permeability'] <= -4)]
print(data.shape)

# drop NA SMILES
data.dropna(subset=['SMILES'], inplace=True)
print(data.shape)



# SMILES有多个合法方式，故用InChiKey去重
def smiles_to_ick(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ick = Chem.MolToInchi(mol)
    return ick

# apply smiles_to_ick
data['InChIKey'] = data['SMILES'].apply(smiles_to_ick)
print(data['InChIKey'])
# remove duplicate
data = data.drop_duplicates(subset = 'InChIKey').reset_index(drop=True)
print(data.shape) # 8097 -> 7626

