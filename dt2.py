from numpy import argmax, log2, sqrt, ceil, round, mean
import random as rd
from collections import defaultdict
#Same general logic as HW1 decision tree but modified to make assignment and support for numerical
#attributes
class Node:
    def __init__(self) -> None:
        self.label = None
        self.attribute = None
        self.left = None
        self.right = None
        self.threshold = None


def allLabeled(D, classIndex):
    label = D[0][classIndex]
    for d in D:
        
        if not (d[classIndex] == label):
            return False 
    return True

def majorityLabel(D, classIndex):
    labelCounts = defaultdict(int)
    for d in D:
        label = d[classIndex]
        labelCounts[label] += 1
    maxkey = max(zip(labelCounts.values(), labelCounts.keys()))[1]
    return maxkey

def getProbs(D, classIndex):
    probs = defaultdict(int)
    counts = defaultdict(int)
    count = 0
    for d in D:
        count += 1
        counts[d[classIndex]] += 1
    for label in counts.keys():
        probs[label] = counts[label]/count

    return probs

def entropy(D, classIndex):
    if len(D) == 0:
        return 0
    probs = getProbs(D, classIndex)
    entropy = 0
    for label in probs:
        entropy -= probs[label] * log2(probs[label])
    
    return entropy

def partition(D, attr, threshold):
    lte = []
    gt = []
    for d in D:
        if d[attr] <= threshold:
            lte.append(d)
        else:
            gt.append(d)
    
    return lte,gt
    
def infoGain(D, L, classIndex, m):
    #Randomly sample  attributes without replacement
    l = []
    if len(L) > 4:
        i = 0
        while i < m:
            attr = L[rd.randrange(len(L))]
            if attr in l:
                continue
            else:
                l.append(attr)
                i += 1
    else:
        l = L

    orig = entropy(D, classIndex)
    infogains = defaultdict(int)
    for attr in l:
        #Calculating threshold for numeric splits
        vals = []
        for d in D:
            vals.append(d[attr])
        threshold = mean(vals)
        lte, gt = partition(D, attr, threshold)
        avg = (len(lte)/len(D))*entropy(lte, classIndex) + (len(gt)/len(D))*entropy(gt, classIndex)
        infogains[attr] = (orig - avg, threshold)

    maxattr = max(zip(infogains.values(), infogains.keys()))[1]
    return (maxattr, infogains[maxattr][1])

def decision_tree(D, L, classIndex, m, min_split_size):
    N = Node()
    if(allLabeled(D, classIndex)):
        N.label = D[0][classIndex] 
        return N
    if(not L):
        N.label = majorityLabel(D, classIndex)
        return N
    if(len(D) <= min_split_size):
        N.label = majorityLabel(D, classIndex)
        return N
    
    A = infoGain(D, L, classIndex, m)
    N.attribute = A[0]
    N.threshold = A[1]
    left, right = partition(D, A[0], A[1])
    T_v = Node()
    if not left:
        T_v.label = majorityLabel(D, classIndex)
    else:
        T_v = decision_tree(left, L, classIndex, m, min_split_size)
    U_v = Node()
    if not right:
        U_v.label = majorityLabel(D, classIndex)
    else:
        U_v = decision_tree(right, L, classIndex, m, min_split_size)
    N.left = T_v 
    N.right = U_v
    return N

def testTree(t, d):
    if t.left == None and t.right == None:
        return t.label
    if d[t.attribute] <= t.threshold:
        return testTree(t.left, d)
    else:
        return testTree(t.right, d)
    
    