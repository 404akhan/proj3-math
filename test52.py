from helpers import *

version = 60
betas = pickle.load(open('models/betas-v%d.p'%version, 'rb'))
betas0 = pickle.load(open('models/betas0-v%d.p'%version, 'rb'))
betas_abs = np.abs(betas)

print betas_abs.shape
print betas0.shape

arr = np.mean(betas_abs, axis=0)[-32:] > np.mean(np.mean(betas_abs, axis=0)) + np.std(np.mean(betas_abs, axis=0))

print arr
print arr.shape