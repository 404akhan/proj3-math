import numpy as np
import pandas as pd

df = pd.read_csv('data/train3M_drug_pair.csv')

drug1 = np.array(df['Drug1'])
drug2 = np.array(df['Drug2'])

print drug1
print drug2

name2index = {}
name2index = {} 

for dr in drug1:
	name2index[dr] = 0

for dr in drug2:
	name2index[dr] = 0

counter = 0
for i in name2index:
	name2index[i] = counter
	counter += 1

print counter
print len(name2index)
print name2index

