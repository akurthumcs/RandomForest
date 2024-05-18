from numpy import argmax, log2, mean
import random as rd
V = [0, 1, 2]

class Node:
    def __init__(self) -> None:
        self.label = None
        self.attribute = None
        self.children = []
    

def allLabeled(D):
    label = D[0][16]
    for d in D:
        if not d[16] == label:
            return False 
    return True

def majorityLabel(D):
    labelCounts = [0, 0]
    for d in D:
        label = d[16]
        labelCounts[label] += 1
    return argmax(labelCounts)

def getProb(label, part):
    count = 0
    for elem in part:
        if elem[16] == label:
            count += 1
    return count/len(part)
        

def entropy(D):
    if len(D) == 0:
        return 0
    p0 = getProb(0, D)
    p1 = 1 - p0
    if(p0 == 0 or p1 == 0):
        return 0
    return (-p0 * log2(p0)) + (-p1 * log2(p1))

def maxAttr(infoGains):
    max = -1
    maxattr = -1
    for tupl in infoGains:
        if tupl[0] > max:
            max = tupl[0]
            maxattr = tupl[1]
    
    return maxattr

def infoGain(D, L, m):
    l = []
    if len(L) > m:
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
    orig = entropy(D)
    infoGains = []
    for attr in l:
        parts = []
        for v in V:
            parts.append(partition(D, attr, v))
        ents = []
        for part in parts:
            ents.append(entropy(part))
        avg = mean(ents)
        infoGains.append((orig - avg, attr))
    return maxAttr(infoGains)

def partition(D, A, v):
    part = []
    for d in D:
        if d[A] == v:
            part.append(d)
    return part

def decision_tree(D, L: list, m, min_split_size):
    N = Node()
    if(allLabeled(D)):
        N.label = D[0][16] 
        return N
    if(not L):
        N.label = majorityLabel(D)
        return N
    if(len(D) <= min_split_size):
        N.label = majorityLabel(D)
        return N
    A = infoGain(D, L, m)
    N.attribute = A
    for v in V:
        D_v = partition(D, A, v)
        T_v = Node()
        if not D_v:
            T_v.label = majorityLabel(D)
        else:
            T_v = decision_tree(D_v, L, m, min_split_size)
        N.children.append((T_v, v))
    return N

def isLeaf(node: Node):
    return len(node.children) == 0

def findBranch(children, val):
    for child in children:
        if child[1] == val:
            return child[0]

#One instance
def testTree(tree: Node, d):
    if isLeaf(tree):
        return tree.label
    else:
        return testTree(findBranch(tree.children, d[tree.attribute]), d)