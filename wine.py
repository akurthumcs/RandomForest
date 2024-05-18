import pandas as pd
import numpy as np
import random
import math
import metrics
from matplotlib import pyplot as plt
from collections import defaultdict
from dt2 import decision_tree, testTree


def bootstrap(D):
    strap = []
    size = len(D)
    for i in range(size):
        j = random.randrange(size)
        strap.append(D[j])
    
    return strap

def randomForest(D, ntree):
    trees = []
    for b in range(ntree):
        strap = bootstrap(D)
        trees.append(decision_tree(strap, ATTS, 0, 4, 2))
    return trees

def k_fold(D, k):
    folds = []
    Dcopy = np.copy(D)
    foldSize = math.ceil(len(D)/k)
    for j in range(k):
        fold = []
        for i in range(foldSize):
            index = random.randrange(len(Dcopy))
            fold.append(Dcopy[index])
            np.delete(Dcopy, index)
        folds.append(fold)
    return folds

def testForest(test, f):
    preds = []
    for d in test:
        predcounts = defaultdict(int)
        for tree in f:
            pred = testTree(tree, d)
            predcounts[pred] += 1
        prediction = max(zip(predcounts.values(), predcounts.keys()))[1]
        preds.append((prediction, d[0]))
    return preds
        
ntrees = [1, 5, 10, 20, 30, 40, 50]
ATTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ,12, 13]
K = 10

wine = pd.read_csv('hw3_wine.csv', delim_whitespace=True, float_precision=None)
wine = wine.to_numpy()

accuracies = []
recalls = []
precisions = []
f1s = []
for ntree in ntrees:
    kaccs = []
    kprecisions = []
    krecalls = []
    kfscores = []
    folds = k_fold(wine, K)
    for k in range(K):
        #Add all folds to training except fold k
        train = []
        for i in range(K):
            if i == k:
                continue
            for j in range(len(folds[i])):
                train.append(folds[i][j])
    
        forest = randomForest(train, ntree)
        preds = testForest(folds[k], forest)
        cmatrix = metrics.confusionMatrix(preds)
        acc = metrics.accuracy(cmatrix, preds)
        kaccs.append(acc)
        precision = metrics.precision(cmatrix)
        kprecisions.append(precision)
        recall = metrics.recall(cmatrix)
        krecalls.append(recall)
        fscore = metrics.fscore(recall, precision, 1)
        kfscores.append(fscore)
    
    accuracies.append(np.mean(kaccs))
    precisions.append(np.mean(kprecisions))
    recalls.append(np.mean(krecalls))
    f1s.append(np.mean(kfscores))

print(kfscores)
plt.plot(ntrees, accuracies)
plt.show()

plt.plot(ntrees, precisions)
plt.show()

plt.plot(ntrees, recalls)
plt.show()

plt.plot(ntrees, f1s)
plt.show()



        

        

    