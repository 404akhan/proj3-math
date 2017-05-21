# submit

from helpers import *
import matplotlib.pyplot as plt

version = 500
train, test = get_tr_cv()
cells = pickle.load(open('data/cells_final_combo.p', 'rb')) 
labels = get_labels_2()
betas = pickle.load(open('models/betas-v%d.p'%version, 'rb'))
betas0 = pickle.load(open('models/betas0-v%d.p'%version, 'rb'))
print train.shape
print cells.shape
print betas.shape
print betas0.shape
print np.sum(labels==0), np.sum(labels==1), np.sum(labels==labels)
print version

# prediction
df = pd.read_csv('data/train3M_drug_pair.csv')
data = np.array(df)

name2index_cell = get_name2index_cell()
name2index_drug = get_name2index_drug()

f1=open('./submission-v5.6M_drug_pair.csv', 'w+')
print >>f1, '"SampleID","ComparisonValue"'
# end prediction

for iteration, sample in enumerate(data):
	name_k, name_i, name_j, target = sample

	# prediction
	if not np.isnan(target): continue
	k = name2index_cell[name_k]
	i = name2index_drug[name_i]
	j = name2index_drug[name_j]
	# end # prediction

	y = np.tanh( 
		np.dot(betas[labels[k], i, :], cells[k, :]) - np.dot(betas[labels[k], j, :], cells[k, :]) + \
			betas0[labels[k], i] - betas0[labels[k], j]
	)
	predict = 1 if y > 0 else -1

	print >>f1, '"%s",%d' % (iteration+1, predict)

print 'end of program'

