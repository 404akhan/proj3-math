# special k top sort

from helpers import *

total = 988
pairs_train = pickle.load(open('data/pairs_train-Kall.p', 'rb'))
pairs_test = pickle.load(open('data/pairs_test-Kall.p', 'rb'))
num = 265

for k in range(total):
	last = np.mean(pairs_train[k])
	print '\nstarting k', k, 'starting value', last
	for time in range(10):
		for i in range(num):
			for j in range(num):
				if pairs_train[k][i][j] == 1:
					pairs_train[k][i, :] = (pairs_train[k][i, :]+pairs_train[k][j, :] > 0).astype(int)
		new = np.mean(pairs_train[k])
		print time, new	
		
		if last == new:
			break
		else:
			last = new

# testing 
correct = 0
incorrect = 0
total_prediction = 0
for k in range(total):
	if k % 100 == 0: print k
	for i in range(num):
		for j in range(num):
			if pairs_test[k][i][j] == 1:
				total_prediction += 1
				if pairs_train[k][i][j] == 1:
					correct += 1
				if pairs_train[k][j][i] == 1:
					incorrect += 1

print 'testing results', correct, incorrect, total_prediction

pickle.dump(pairs_train, open('data/pairs_train-Kall-complete.p', 'wb'))



sys.exit()
num = 265
graph = {k: [] for k in range(265)}
for i in range(265):
	for j in range(265):
		if pairs_train[i][j] == 1:
			graph[i].append(j)

for times in range(num+1):
	for i in range(num):
		neighbors = copy.deepcopy(graph[i])
		for j in neighbors:
			for k in graph[j]:
				if k not in graph[i]:
					graph[i].append(k)


sys.exit()

topol_sorted = []
def dfs_rec(graph,start,path):
    path = path + [start]
    for edge in graph[start]: 
        if edge not in path:
            path = dfs_rec(graph, edge,path)
    topol_sorted.append(start)
    return path

dfs_rec(graph, 0, [])
topol_sorted = topol_sorted[::-1]

print len(topol_sorted), topol_sorted

pairs_enlarged = np.zeros((num, num))
for i in range(len(topol_sorted)):
	for j in range(i+1, len(topol_sorted)):
		pairs_enlarged[topol_sorted[i]][topol_sorted[j]] = 1

print pairs_enlarged

correct = 0
incorrect = 0
for i in range(num):
	for j in range(num):
		if pairs_test[i][j] == 1 and pairs_enlarged[i][j] == 1:
			correct += 1
		if pairs_test[i][j] == 1 and pairs_enlarged[j][i] == 1:
			incorrect += 1

print correct, incorrect