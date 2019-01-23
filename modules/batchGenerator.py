import math
import numpy as np
from modules.wordsToVectors import wordsToVectors

def createTrainBatchGenerator(trainData, embeddings, batchSize):
    numberOfBatches = math.ceil(len(trainData) / batchSize)
    while True: 
        trainData = trainData.sample(frac=1.)  # Shuffle the data.
        for i in range(numberOfBatches):
            questions = trainData.iloc[i * batchSize : (i + 1) * batchSize, 1]
            questionsArray = np.array(wordsToVectors(embeddings, questions))
            yield questionsArray, np.array(trainData["target"][i * batchSize:(i + 1) * batchSize])


def createPredictionBatchGenerator(testData, embeddings, batchSize):
    numberOfBatches = math.ceil(len(testData) / batchSize)

    for i in range(numberOfBatches):
        questions = testData.iloc[i * batchSize : (i + 1) * batchSize, 1]
        questionsArray = np.array(wordsToVectors(embeddings, questions))
        yield questionsArray
