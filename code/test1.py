import numpy as np
import pandas as pd

df = pd.read_csv('data/train3M_drug_pair.csv')

comp = np.array(df['Comparison'])

print comp

c1 = 0
c2 = 0
c3 = 0
c4 = 0

for i in range(len(comp)):
	if comp[i] == 1:
		c1 += 1
	elif comp[i] == -1:
		c2 += 1
	elif np.isnan(comp[i]):
		c3 += 1
	else:
		c4 += 1

print c1, c2, c3, c4
