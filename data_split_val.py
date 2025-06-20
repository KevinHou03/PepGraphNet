import pandas as pd
import numpy as np

test_train = pd.read_csv('/Users/kevinhou/Documents/CY Lab/Data/lariat_caco2/lariat_caco2_train.csv')
test_val = pd.read_csv('/Users/kevinhou/Documents/CY Lab/Data/lariat_caco2/lariat_caco2_test.csv')

print(test_train.shape)
print(test_val.shape)