import numpy as np
import pandas_datareader as web
import datetime as dt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler

"""Important Global variables
string ticker: ticker symbol used to get historical data of that company.
int featureNo: default to 4 as the amount of features used are 'High', 'Low', 'Open', 'Adj Close'. 
int predictionDays: amount of previous day data used to make the next day prediction. Value is a hyperParamer. 
"""
ticker = "FB"
featureNo = 4
predictionDays = 7

"""Train Data Dates(yyyy, mm, dd)"""
trainStart = dt.datetime(2012, 1, 1)
trainEnd = dt.datetime(2021, 4, 1)
"""Test Data Dates(yyyy, mm, dd)"""
TestStart = dt.datetime(2021, 4, 2)
TestEnd = dt.datetime(2022, 4, 1)

def getTrainData():
    """Uses pandas_datareader to read historical data from yahoo finance to use as the train data. 
    Also, creates a standard scaler using sklearn to make it easier for the model.
    return: Matrix of scaled down data, the scaler used to scale the data.
    """
    data = web.DataReader(ticker, "yahoo", trainStart, trainEnd)
    # The scaler is fit using the value of the features listed and used to scale the data.
    scaler = StandardScaler().fit(data[["High", "Low", "Open", "Adj Close"]].values)
    scaledData = scaler.transform(data[["High", "Low", "Open", "Adj Close"]].values)
    return scaledData, scaler

def getTestData(scaler):
    """Uses pandas_datareader to read historical data from yahoo finance to use as the test data.
    scaler is input to scale the test data as done to the train data.
    return: Scaled down matrix of historical data, matrix of actual historical data.
    """
    data = web.DataReader(ticker, "yahoo", TestStart, TestEnd)
    actualPrices = data[["High", "Low", "Open", "Adj Close"]]
    actualPrices = np.array(actualPrices)

    data = scaler.transform(actualPrices)
    return data, actualPrices

def formatTrainData(importedTrainData):
    """Pre-processing of train data to format it correctly for machine learning.
    Matrix/List importedTrainData train data that will be used as machine learning.
    return: Matrix/List trainX the correctly formated inputs that will be used for the nural network,
    Matrix/List trainY the correctly formated data input that will be used when error checking within machine learning.
    """
    trainX = []
    trainY = []
    # loop starts from predictionDays to account for negative values when trying to retrive train data.
    for i in range(predictionDays, len(importedTrainData)):
        #list of values appened to trainX ranging from start of prediction days to end of prediction days.
        trainX.append(importedTrainData[i - predictionDays: i])
        #value of the day, after prediction days, appened to trainY.
        trainY.append(importedTrainData[i, 0: featureNo])

    trainX, trainY = np.array(trainX), np.array(trainY)
    return trainX, trainY

def formatTestData(importedTestData):
    """Pre-processing of test data to format it correctly for machine learning.
    Matrix/List importedTestData test data that will be used as the input when predicting, post learning.
    return: Matrix/List testX the correctly formated data that'll be used to make predictions.
    """
    testData = importedTestData

    testX = []
    #loop starts from predictionDays to account for negative values when trying to retrive test data.
    for i in range (predictionDays, len(testData)):
        #list of values appened to testX ranging from start of prediction days to end of prediction days.
        testX.append(testData[i - predictionDays: i])

    testX = np.array(testX)

    return testX

def createModel(layerAmount, unitAmount, epochAmount, batchSize_used, dropoutAmount, xInput, yInput):
    """Builds and trains a neural network model using the imported hyperparameters and train data.
    int layerAmount the amount of layers the model will have,
    int unitAmount the amount of units each layer will have,
    int epochAmount the amount of epochs that will be run for the model to train,
    int batchSize_used the bactch size that will be used when training,
    float dropoutAmount the amount of dropout that will be used int the dropout layers,
    Matrix/List xInput the inputs that are used to train the neural network,
    Matrix/List yInput the inputs that will be used to error check the neural network whilst training.
    return: Seqential model the neural network model that was trained using the x and y inputs with the set 
    hyperparameters.
    """
    #Determines whether the layer, to be added, is the last layer of the neural network.
    #If the layer is the last layer then the return_sequences is set to false.
    if(layerAmount == 0):
        #model is created and a LSTM and dropout layer is added to the model.
        model = Sequential()
        model.add(LSTM(units = unitAmount, return_sequences = False, input_shape = (xInput.shape[1], xInput.shape[2])))
        model.add(Dropout(dropoutAmount))
    else:
        model = Sequential()
        model.add(LSTM(units = unitAmount, return_sequences = True, input_shape = (xInput.shape[1], xInput.shape[2])))
        model.add(Dropout(dropoutAmount))        

    startLayerAmount = layerAmount
    #creates layers for the amount of layers specified in the parameters.
    for i in range(0, startLayerAmount):
        if(layerAmount == 1):
            model.add(LSTM(units = unitAmount, return_sequences = False,))
            model.add(Dropout(dropoutAmount))
        else:
            model.add(LSTM(units = unitAmount, return_sequences = True,))
            model.add(Dropout(dropoutAmount))
        layerAmount -= 1
    #Dense layer is added specifying the amount of units is the shape of the yInput
    model.add(Dense(units = yInput.shape[1]))
    #Model is complied and fit so the weights are calculated and the network is trained.
    model.compile(optimizer = "adam", loss = "mse")
    model.fit(xInput, yInput, epochs = epochAmount, batch_size = batchSize_used, verbose = 0)
    return model

def hyperPerameterTweaking(maxLayers, maxUnits, uDecriment, maxEpochs, eDecriment, maxBatchSize, BSDecriment, maxDropout):
    """Loops through different amounts of hyperparamters and creates a neural network which is accuracy checked.
    int maxLayers largest amount of layers that will be used when creating the model,
    int maxUnits largest amount of units that will be used when creating the model,
    int uDecriment amount that is decrimented, from the units used, when looping,
    int maxEpochs largest amount of epochs that will be used when creating the model,
    int eDecriment amount that is decrimented, from the epochs used, when looping,
    int maxBatchSize largest batchSize that will be used when creating the model,
    int BSDecriment amount that is decrimented, from the batachSize used, when looping,
    float maxDropout amount of dropout used when creating the model.
    Return: dictionary predictionAccuracyScores calculated accuracy scores of the predictions made by the model,
    dictionary models holds the values of the layers,units,epochs, and batchSize of the different models used.
    """
    layers = maxLayers
    unitDecriment = uDecriment
    epochDecriment = eDecriment
    batchSizeDecriment = BSDecriment
    dropout = maxDropout
    #Fetches and formats train data
    trainData, scaler = getTrainData()
    trainX, trainY = formatTrainData(trainData)
    #Fetches and formats test data
    testX, actualPrices = getTestData(scaler)
    formattedTestX = formatTestData(testX)

    predictionAccuracyScores = {}
    models = {}
    modelNo = 0
    #Loops through hyperparameters whilst decrmenting them by their respected amounts.
    while layers >= 0:
        units = maxUnits
        while units >= unitDecriment:
            epochs = maxEpochs
            while epochs >= epochDecriment:
                batchSize = maxBatchSize
                while batchSize >= batchSizeDecriment:
                    #For each loop a model is created using the corisponding hyperparameters,
                    #A prediction is made, using that model, and unscaled.
                    model = createModel(layers, units, epochs, batchSize, dropout, trainX, trainY)
                    predictedPrices = model.predict(formattedTestX)
                    predictedPrices = scaler.inverse_transform(predictedPrices)
                    #Test accuracy fetched and stored in predictionAccuracyScores dictionary under the key of an 
                    #incrimenting model number. Hyperparameters are stored in the models dictionary under the same key.
                    percentageError = testAccuracy(predictedPrices, actualPrices)
                    predictionAccuracyScores.update({f"model {modelNo}": percentageError})
                    models.update({f"model {modelNo}": {"layers" : layers, "units" : units, "epochs" : epochs, "batchSize" : batchSize}})
                    modelNo += 1
                    batchSize -= batchSizeDecriment
                epochs -= epochDecriment
            units -= unitDecriment
        layers -= 1
                    
    return predictionAccuracyScores, models

def testAccuracy(predictedPrices, actualPrices):
    """Calculates the percentage error using the the predicted prices and actual prices of the stock.
    Matrix/List predictedPrices prices that the model predicted,
    Matrix/List actualPrices actual prices of the test data.
    Return: float percentageError absolute value of the percentage error calculated
    """
    actualPrices = actualPrices
    #Deletes first few days of actual prices that was used to make the first prediction.
    for i in range(0, predictionDays):
        actualPrices = np.delete(actualPrices, 0, 0)
    #Sums up all the values in the Matrix/Lists
    predSum = np.sum(predictedPrices)
    actSum = np.sum(actualPrices)
    #Uses percentage error formula.
    percentageError = np.subtract(predSum, actSum)
    percentageError = np.divide(percentageError, actSum)
    percentageError = np.multiply(percentageError, 100)
    return abs(percentageError)

def sortDict(dict):
    """Sorts dictionaries by acending order of their values.
    Dictionary dict the dictionary that will be sorted.
    Return: Dictionary sortedDict a dictionary sorted by acending values"""
    sortedDict = {k: v for k, v in sorted(dict.items(), key = lambda item: item[1])}
    return sortedDict

def dictAddition(cumulativeDict, percErrorDict):
    """Adds together the values within two dictionaries
    Dictionary cumulativeDict main cumulative dictionary of percentage errors,
    Dictionary percErrorDict the dictionary containing the current percentage errors calculated"""
    #Checks to see if main dictionary is empty if so no addition is required.
    if(cumulativeDict):
        for x, y in cumulativeDict.items():
            #Key stays the same. values get added.
            cumulativeDict.update({x : y + percErrorDict[x]})
    else:
        cumulativeDict = percErrorDict
    return cumulativeDict

def calcMean(cumulativeDict, loopAmount):
    """Calculates the mean of the main dictionary containing all percentage errors added.
    Dictionary cumulativeDict main cumulative dictionary of percentage errors
    Int loopAmount the amount of loops of hyperparam tweaking that occured"""
    #Loops through all items in cumulativeDict dividing their values by loopAmount
    for x, y in cumulativeDict.items():
        cumulativeDict.update({x : y / loopAmount})
    return cumulativeDict

"""Hyper Params to be tweaked for testing (Read Me... for more info)"""
#Max number of layers that can be used      0-5(aprx recomended values for default data set)
maxLayers = 1
#Max number of units used in network        2-2580(aprx recomended values for default data set)
maxUnits = 16
#Amount the units decriment each loop       (No recomended amount)
unitDecriment = 16
#Max amount of days that should be used for prediction       (No recomended amount)
maxPredictionDays = 7
#Max amount of epochs that should be used for prediction       1-128(aprx recomended values for default data set)
maxEpochs = 32
#Amount epochs decrease with each loop     (No recomended amount)
epochDecriment = 16
#Max amount size of batches that should be used for predicition        2-2580(aprx recomended values for default data set)
maxBatchSize = 32
#Amount epochs decrease with each loop     (No recomended amount)
batchSizeDecriment = 16
#Max dropout used for dropout layers       0.2-0.8(aprx recomended values for default data set)
maxDropout = 0.2
#Ammount of times hyperParamTweaking should loop to give multiple attempts with each model to account for anomalies
loopAmount = 5 
sumAccuracyScores = {}

for i in range(0, loopAmount):
    #Hyperparameter tweak and sums all loops of this into one cumulative dictionary.
    predictionAccuracyScores, models = hyperPerameterTweaking(maxLayers, maxUnits, unitDecriment, maxEpochs, epochDecriment, maxBatchSize, batchSizeDecriment, maxDropout)
    sumAccuracyScores = dictAddition(sumAccuracyScores, predictionAccuracyScores)
#Finds the mean of the cumulative dictionary and sorts it in accending order
meanAccuracyScores = calcMean(sumAccuracyScores, loopAmount)
meanAccuracyScores_sorted = sortDict(meanAccuracyScores)
#Model with lowest percentage error is then saved to a variable which is printed.
bestModel, percentageError = list(meanAccuracyScores_sorted.items())[0]
#print(meanAccuracyScores_sorted) # Prints all the mean accuracy scores if wanted to be viewed (lower value is better)
print(f"The best hyperParameters for this stock is {bestModel}:{models[bestModel]}\nIt has a percentage error of {percentageError}")