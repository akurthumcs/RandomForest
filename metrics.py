import numpy as np 
    #    +   -   -     
    # +  TP  FP  FP
    # -  FN  TN  TN
    # -  FN  TN  TN

    #    -   +   -     
    # -  TN  FN  TN
    # +  FP  TP  FP
    # -  TN  FN  TN

    #    -   -   +     
    # -  TN  TN  FN
    # -  TN  TN  FN
    # +  FP  FP  TP  
def confusionMatrix(preds):
    cmatrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for pred in preds:
        cmatrix[int(pred[0]) - 1][int(pred[1]) - 1] += 1
    return cmatrix

def accuracy(cmatrix, preds):
    n = len(preds)
    accsum = 0
    accsum += (cmatrix[0][0] + cmatrix[1][1] + cmatrix[1][2] + cmatrix[2][1] + cmatrix[2][2])/n
    accsum += (cmatrix[1][1] + cmatrix[0][0] + cmatrix[0][2] + cmatrix[2][0] + cmatrix[2][2])/n
    accsum += (cmatrix[2][2] + cmatrix[0][0] + cmatrix[0][1] + cmatrix[1][0] + cmatrix[1][1])/n
    return accsum/3

def precision(cmatrix):
    precisionsum = 0
    precisionsum += cmatrix[0][0] / (cmatrix[0][0] + cmatrix[0][1] + cmatrix[0][2])
    precisionsum += cmatrix[1][1] / (cmatrix[1][1] + cmatrix[1][0] + cmatrix[1][2])
    precisionsum += cmatrix[2][2] / (cmatrix[2][2] + cmatrix[2][0] + cmatrix[2][1])
    return precisionsum/3 

def recall(cmatrix):
    recallsum = 0
    recallsum += cmatrix[0][0]/ (cmatrix[0][0] + cmatrix[1][0] + cmatrix[2][0])
    recallsum += cmatrix[1][1]/ (cmatrix[1][1] + cmatrix[0][1] + cmatrix[2][1])
    recallsum += cmatrix[2][2]/ (cmatrix[2][2] + cmatrix[0][2] + cmatrix[1][2])
    return recallsum/3

def fscore(recall, precision, beta):
    return 2 * ((recall * precision)/(recall + precision))