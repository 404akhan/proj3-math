# predicting

from helpers import *
import matplotlib.pyplot as plt

cells = pickle.load(open('data/very_new_cells-70.p', 'rb')) 
print cells.shape

version = 20
betas = pickle.load(open('models/betas-v%d.p'%version, 'rb'))
betas0 = pickle.load(open('models/betas0-v%d.p'%version, 'rb'))

df = pd.read_csv('data/train3M_drug_pair.csv')
data = np.array(df)

name2index_cell = get_name2index_cell()
name2index_drug = get_name2index_drug()

f1=open('./submission-v2.6M_drug_pair.csv', 'w+')
print >>f1, '"SampleID","ComparisonValue"'

print data
for iteration, sample in enumerate(data):
	name_k, name_i, name_j, target = sample
	if not np.isnan(target): continue

	k = name2index_cell[name_k]
	i = name2index_drug[name_i]
	j = name2index_drug[name_j]

	y = np.tanh( 
		np.dot(betas[i, :], cells[k, :]) - np.dot(betas[j, :], cells[k, :]) + betas0[i] - betas0[j]
	)

	predict = 1 if y > 0 else -1
	print >>f1, '"%s",%d' % (iteration+1, predict)




print 'end of program'

sys.exit()
correct = 0
total = 0
for iteration, sample in enumerate(test):
	k, i, j, target = sample

	y = np.tanh( 
		np.dot(betas[i, :], cells[k, :]) - np.dot(betas[j, :], cells[k, :]) + betas0[i] - betas0[j]
	)

	predict = 1 if y > 0 else -1
	if predict == target:
		correct += 1
	total += 1

print correct * 1.0 / total
