# see when errors according to certainty from matrix if there use average over users that same cancer

from helpers import *

df = pd.read_csv('data/train3M_drug_pair.csv')
train, test = get_tr_cv()
drug_num = get_total_drug()
pairs = np.zeros((drug_num, drug_num))
total_comp = np.eye(drug_num, drug_num)

for sample in train:
	k, i, j, target = sample
	if target == 1:	pairs[i][j] += 1
	if target == -1: pairs[j][i] += 1

	total_comp[i][j] += 1
	total_comp[j][i] += 1
confidence = pairs*1. / total_comp

# adding gen feautere linear model
cells = pickle.load(open('data/cells_final_combo.p', 'rb')) 
labels = get_labels_2()
version = 112
betas = pickle.load(open('models/betas-v%d.p'%version, 'rb'))
betas0 = pickle.load(open('models/betas0-v%d.p'%version, 'rb'))
# end. adding gen feautere linear model

### testing
error1 = []
error2 = []
correct = []
count_uncertain = 0
for sample_test in test:
	k, i, j, target = sample_test

	### linear
	y = np.tanh( 
		np.dot(betas[labels[k], i, :], cells[k, :]) - np.dot(betas[labels[k], j, :], cells[k, :]) + \
			betas0[labels[k], i] - betas0[labels[k], j]
	)
	predict = 1 if y > 0 else -1
	### end. linear
	### if uncertain
	if max(confidence[i][j], confidence[j][i]) < 0.35:
		cells_aver = np.mean(cells, axis=0)
		y = np.tanh( 	
			np.dot(betas[labels[k], i, :], cells_aver) - np.dot(betas[labels[k], j, :], cells_aver) + \
				betas0[labels[k], i] - betas0[labels[k], j]
		)
		predict = 1 if y > 0 else -1
		count_uncertain += 1
	### end uncertain
	
	if predict == 1 and target != 1:
		error1.append(confidence[i][j])
	if predict == -1 and target != -1:
		error2.append(confidence[j][i])
	if predict == target:
		correct.append(max(confidence[i][j], confidence[j][i]))

print 'error1', len(error1), np.mean(error1), np.std(error1)
print 'error2', len(error2), np.mean(error2), np.std(error2)
print 'correct', len(correct), np.mean(correct), np.std(correct)
print 'total_uncertain', count_uncertain