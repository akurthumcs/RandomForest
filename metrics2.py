def confusionMatrix(results):
    cMatrix = [[0, 0], [0, 0]]
    for d in results:
        cMatrix[d[0]][d[1]] += 1
    
    return cMatrix

def accuracy(cMatrix, results):
    return (cMatrix[0][0] + cMatrix[1][1]) / len(results)

def recall(cMatrix):
    return (cMatrix[0][0])/ (cMatrix[0][0] + cMatrix[1][0])

def precision(cMatrix):
    return (cMatrix[0][0])/ (cMatrix[0][0] + cMatrix[0][1])

def fscore(recall, precision, beta):
    return 2 * ((recall * precision)/(recall + precision))