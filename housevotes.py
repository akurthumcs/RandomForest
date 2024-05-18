import random
import pandas as pd
import dt
import numpy as np
import math
from collections import defaultdict
import metrics2 as m
from matplotlib import pyplot as plt

ATTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] 
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
        trees.append(dt.decision_tree(strap, ATTS, 4, 100))
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
            pred = dt.testTree(tree, d)
            predcounts[pred] += 1
        prediction = max(zip(predcounts.values(), predcounts.keys()))[1]
        preds.append((prediction, d[16]))
    return preds


ntrees = [1, 5, 10, 20, 30, 40, 50]
K = 10

housevotes = pd.read_csv('hw3_house_votes_84.csv')

housevotes = housevotes.to_numpy()
accuracies = []
recalls = []
precisions = []
f1s = []
for ntree in ntrees:
    kaccs = []
    kprecisions = []
    krecalls = []
    kfscores = []
    folds = k_fold(housevotes, K)
    for k in range(K):
        train = []
        for i in range(K):
            if i == k:
                continue
            for j in range(len(folds[i])):
                train.append(folds[i][j])
        test = folds[k]
    
        forest = randomForest(train, ntree)
        preds = testForest(test, forest)
        cMatrix = m.confusionMatrix(preds)
        acc = m.accuracy(cMatrix, preds)
        kaccs.append(acc)
        precision = m.precision(cMatrix)
        kprecisions.append(precision)
        recall = m.recall(cMatrix)
        krecalls.append(recall)
        f1 = m.fscore(recall, precision, 1)
        kfscores.append(f1)
    
    accuracies.append(np.mean(kaccs))
    recalls.append(np.mean(krecalls))
    precisions.append(np.mean(kprecisions))
    f1s.append(np.mean(kfscores))


plt.plot(ntrees, accuracies)
plt.show()

plt.plot(ntrees, recalls)
plt.show()

plt.plot(ntrees, precisions)
plt.show()

plt.plot(ntrees, f1s)
plt.show()
    
        


