from helpers import *

df = pd.read_csv('data/train3M_drug_pair.csv')

print df

train, test = get_tr_cv()

print train
print train.shape

drug_num = get_total_drug()

print drug_num
pairs = np.zeros((drug_num, drug_num))

total_comp = np.eye(drug_num, drug_num)

for sample in train:
	k, i, j, target = sample
	pairs[i][j] += target
	pairs[j][i] -= target

	total_comp[i][j] += 1
	total_comp[j][i] += 1

print pairs
print total_comp
print pairs*1. / total_comp

confidence = np.abs(pairs*1. / total_comp)

print confidence > 0.95
print np.sum(confidence > 0.99), np.sum(confidence > 0.99)*1.0 / np.prod(confidence.shape)
print np.sum(confidence > 0.95), np.sum(confidence > 0.95)*1.0 / np.prod(confidence.shape)
print np.sum(confidence > 0.9), np.sum(confidence > 0.9)*1.0 / np.prod(confidence.shape)

### testing
threshold = 0.8
confidence_signed = pairs*1. / total_comp

correct_predicted = 0
total_predicted = 0
for sample_test in test:
	k, i, j, target = sample_test
	predict = 0
	if confidence_signed[i][j] > threshold: predict = 1
	if confidence_signed[i][j] < -threshold: predict = -1

	if predict != 0:
		correct_predicted += 1 if predict == target else 0
		total_predicted += 1
print 'threshold used %.2f' % threshold 
print correct_predicted*1.0 / total_predicted, \
	'out of total_predict %d, test_size %d'%(total_predicted, len(test))
