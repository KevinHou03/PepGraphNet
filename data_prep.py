import ast
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

# load data
data = pd.read_csv('/Users/kevinhou/Documents/CY Lab/Data/CycPeptMPDB_Peptide_All (1).csv', low_memory=False) # this address mixed datatype
print(data.shape) # 8466, 247

'''
1. "data WITHOUT permeability values or assay methods were removed"
'''
data.dropna(subset=['Permeability'], inplace=True) # Same_Peptides_Permeability or Permeability?
# data.dropna(subset=['Same_Peptides_Assay'], inplace=True)
data.dropna(subset=['SMILES'], inplace=True)


'''
2. "Data with permeability values less than 10−8 cm/s or greater than10−4 cm/s were 
removed because of their potential unreliability."
'''
data = data[(data['Permeability'] >= -8) & (data['Permeability'] <= -4)]

'''
3. "Removing duplicates within different subsets by calculating the InChiKey to ensure 
data uniqueness" -> SMILES有多个合法方式，故用InChiKey去重
'''
# check if SMILES are all legal
def check_smiles(smiles):
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError('Invalid SMILES')
data['SMILES'].apply(lambda x: check_smiles(x))

def smiles_to_ick(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ick = Chem.MolToInchiKey(mol)
    return ick
# apply smiles_to_ick
data['InChIKey'] = data['SMILES'].apply(smiles_to_ick)
# remove  duplicate
data = data.drop_duplicates(subset = 'InChIKey').reset_index(drop=True) # 8097 -> 7626
print(data.shape)

'''
4. count Assays
'''
#1. assays are str, convert them to python lists
data['Assay_toList'] = data['Same_Peptides_Assay'].apply(lambda x:ast.literal_eval(x) if pd.notna(x) else [])#use pd.notna instead of np.isnan()

#2. list all assays:
assay_set = set()
for row in data['Assay_toList']:
    for item in row:
        if isinstance(item, list):
            assay_set.update(item)
        if isinstance(item, str):
            assay_set.add(item)
print(assay_set) # all assays: {'RRCK', 'Caco2', 'MDCK', 'PAMPA'}
#3. assay count:
assay_count = data[['RRCK', 'Caco2', 'MDCK', 'PAMPA']].notna().sum()
print(assay_count)
'''
RRCK      162
Caco2     918
MDCK       62
PAMPA    6847
dtype: int64
'''
#4. Lariat vs Circle
# 4.1: lariat Caco-2
lariat_caco2 = ((data['Caco2'].notna()) & (data['Molecule_Shape'] == 'Lariat')).sum()
print(lariat_caco2) # 307
circle_caco2 = ((data['Caco2'].notna()) & (data['Molecule_Shape'] == 'Circle')).sum()
print(circle_caco2) # 611
lariat_pampa = ((data['PAMPA'].notna()) & (data['Molecule_Shape'] == 'Lariat')).sum()
print(lariat_pampa) # 2287
circle_pampa = ((data['PAMPA'].notna()) & (data['Molecule_Shape'] == 'Circle')).sum()
print(circle_pampa) # 4560
lariat_rrck = ((data['RRCK'].notna()) & (data['Molecule_Shape'] == 'Lariat')).sum()
print(lariat_rrck) # 0
circle_rrck = ((data['RRCK'].notna()) & (data['Molecule_Shape'] == 'Circle')).sum()
print(circle_rrck)# 162
lariat_mdck = ((data['MDCK'].notna()) & (data['Molecule_Shape'] == 'Lariat')).sum()
print(lariat_rrck)# 0
circle_mdck = ((data['MDCK'].notna()) & (data['Molecule_Shape'] == 'Circle')).sum()
print(circle_mdck)# 53



# data.to_csv('/Users/kevinhou/Documents/CY Lab/Data/CycPeptMPDB_Peptide_Cleaned.csv')
