# ensemble matrix with my gen features, threshold hyperparam

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
cells = pickle.load(open('data/very_new_cells-70.p', 'rb')) 
version = 20
betas = pickle.load(open('models/betas-v%d.p'%version, 'rb'))
betas0 = pickle.load(open('models/betas0-v%d.p'%version, 'rb'))
# end. adding gen feautere linear model

### testing
threshold = 0.95
correct_predicted = 0
total_predicted = 0
for sample_test in test:
	k, i, j, target = sample_test
	predict = 0
	if confidence[i][j] > threshold: predict = 1
	if confidence[j][i] > threshold: predict = -1

	### linear
	y = np.tanh( 	
		np.dot(betas[i, :], cells[k, :]) - np.dot(betas[j, :], cells[k, :]) + betas0[i] - betas0[j]
	)
	if predict == 0: 
		predict = 1 if y > 0 else -1
	### end. linear
	
	if predict != 0:
		correct_predicted += 1 if predict == target else 0
		total_predicted += 1
print 'threshold used %.2f' % threshold 
print correct_predicted*1.0 / total_predicted, \
	'out of total_predict %d, test_size %d'%(total_predicted, len(test))
