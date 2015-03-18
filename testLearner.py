from KNNLearner import KNNLearner
from LinRegLearner import LinRegLearner
import sys
import math
import csv
import matplotlib.pyplot as plotLib
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


def createScatterPlot(xLabel, yLabel, xData, yData, filename):
    plotLib.clf()
    plotLib.plot(xData, yData, 'o')
    plotLib.xlabel(xLabel)
    plotLib.ylabel(yLabel)
    plotLib.savefig(filename, format='pdf')


def createComparisonPlot(xLabel, yLabel, xData, y1Data, y2Data, filename, linename):
    plotLib.clf()
    plotLib.figure().add_subplot(111)
    plotLib.plot(xData, y1Data)
    plotLib.plot(xData, y2Data)
    plotLib.legend(linename)
    plotLib.xlabel(xLabel)
    plotLib.ylabel(yLabel)
    plotLib.savefig(filename, format='pdf')


def calCorrcoef(Y, Ytest):
    return np.corrcoef(Y, Ytest)[0,1]
    
    

if __name__=="__main__":
    filename = sys.argv[1]
    reader = csv.reader(open(filename, 'rU'), delimiter=',')
    count = 0
    for row in reader:
        count = count + 1
    train = int(count * 0.6)
    Xtrain = np.zeros([train,2])
    Ytrain = np.zeros([train,1])
    Xtest = np.zeros([count-train,2])
    Ytest = np.zeros([count-train,1])

    count = 0
    reader = csv.reader(open(filename, 'rU'), delimiter=',')
    for row in reader:
        if(count < train):
            Xtrain[count,0] = row[0]
            Xtrain[count,1] = row[1]
            Ytrain[count,0] = row[2]
            count = count + 1
        else:
            Xtest[count-train,0] = row[0]
            Xtest[count-train,1] = row[1]
            Ytest[count-train,0] = row[2]
            count = count + 1
    
    Y = Ytest[:,0]
    sampleY = Ytrain[:,0]
    bestY = np.zeros([Ytest.shape[0]])
    
    correlation = np.zeros([50])
    gradientRmsError = np.zeros([100])
    rmsError = np.zeros([50])
    kArray = np.zeros([50])
    inSampleRmsErr = np.zeros([50])
    
    #KNN Learner, k vary from 1 to 50
    for k in range(1, 51):
        kArray[k-1] = k
        
        learner = KNNLearner(k)
        learner.addEvidence(Xtrain, Ytrain)
        knnY = learner.query(Xtest)[:,-1]

        #RMS Error(out-of-sample)
        net = 0
        for i in range(0, len(knnY)):
            net = net + (knnY[i] - Y[i]) * (knnY[i] - Y[i])
        knnRMS = math.sqrt(net / len(knnY))

        #In-sample RMS Error
        inSampleTest = learner.query(Xtrain)
        inSampleY = inSampleTest[:,-1]
        net = 0
        for i in range(0, len(inSampleY)):
            net = net + (inSampleY[i] - sampleY[i]) * (inSampleY[i] - sampleY[i])
        insampleRMS = math.sqrt(net / len(inSampleY))

        #Correlation Coefficient
        knnCorr = calCorrcoef(knnY, Y)

        correlation[k-1] = knnCorr
        rmsError[k-1] = knnRMS
        inSampleRmsErr[k-1] = insampleRMS

        if((filename == 'data-classification-prob.csv') and (k == 27)):
            bestY = knnY
        elif((filename == 'data-ripple-prob.csv') and (k == 3)):
            bestY = knnY


    createComparisonPlot('K value', 'RMS Error', kArray, rmsError, inSampleRmsErr, 'comparison.pdf', ['Out-of-Sample Data', 'In-Sample Data'])
    createScatterPlot('Predicted Y', 'Actual Y', bestY, Y, 'bestK.pdf')

    #Linear Regression Learner
    learner = LinRegLearner()
    learner.addEvidence(Xtrain, Ytrain)
    linY = learner.query(Xtest)[:,-1]

    #RMS Error
    net = 0
    for i in range(0, len(linY)):
        net = net + (linY[i] - Y[i]) * (linY[i] - Y[i])
    linRMS = math.sqrt(net / len(linY))
    print linRMS

    #Correlation Coefficient
    linCorr = calCorrcoef(linY, Y)
    print linCorr

    createScatterPlot('Predicted Y', 'Actual Y', linY, Y, 'ScatterLinReg.pdf')