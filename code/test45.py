# tensor

from helpers import *

df = pd.read_csv('data/train3M_drug_pair.csv')
train, test = get_tr_cv()
drug_num = get_total_drug()
pairs = np.zeros((get_total_cancer(), drug_num, drug_num))
total_comp = np.zeros((get_total_cancer(), drug_num, drug_num))
for i in range(drug_num): total_comp[:, i, i] = 1

index2name_cell = get_index2name_cell()
name2index_cancer = get_name2index_cancer()
df_local = pd.read_excel('data/cell_line_features.xlsx')

def map_cell_id2cancer_id_local(cell_id):
	name_cell = index2name_cell[cell_id]
	name_cancer = df_local[name_cell][0]
	return name2index_cancer[name_cancer]

for iteration, sample in enumerate(train):
	k, i, j, target = sample
	cancer_id = map_cell_id2cancer_id_local(k)

	if target == 1:	pairs[cancer_id][i][j] += 1
	if target == -1: pairs[cancer_id][j][i] += 1

	total_comp[cancer_id][i][j] += 1
	total_comp[cancer_id][j][i] += 1

	if iteration % 100*1000==0: print iteration
confidence = pairs*1. / total_comp

print pairs, np.mean(pairs), np.std(pairs)
print total_comp, np.mean(total_comp), np.std(total_comp)

pickle.dump(pairs, open('data/pairs_tensor.p', 'wb'))
pickle.dump(total_comp, open('data/total_comp_tensor.p', 'wb'))