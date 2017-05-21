from helpers import *

df = pd.read_csv('data/train3M_drug_pair.csv')
name2index = get_name2index()
N = len(df)

print N
print df
h1 = df.head(1)

print h1['Drug1'][0]
print h1['Drug2'][0]
print h1['Comparison'][0]

scores = { k : 0 for k in name2index }

for i in range(N):
	if df['Comparison'][i] == 1:
		scores[ df['Drug1'][i] ] += 1
	if df['Comparison'][i] == -1:
		scores[ df['Drug2'][i] ] += 1

	if i % 10000 == 0:
		print i

arr = np.array([v for k, v in scores.iteritems()])
print arr
print np.sum(arr)

print scores

