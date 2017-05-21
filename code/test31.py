from helpers import *

cells = pickle.load(open('data/very_new_cells-70.p', 'rb')) 
betas = pickle.load(open('models/betas-v%d.p'%20, 'rb'))

print cells.shape
print betas.shape

betas_abs = np.abs(betas)
beta_all = np.sum(betas_abs, axis=0)

print beta_all.shape

top = 80
indexes = np.sum(np.argsort(-beta_all)[:1000] < 2450)

print indexes

X = get_matrix_X()
print X.shape

beta_all.sort()
print beta_all[3000:]