import math
import pandas
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors
from modules.wordsToVectors import wordsToVectors

tqdm.pandas()

googleNewsPath = './data/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
trainSetPath = './data/train.csv'

trainSet = pandas.read_csv(trainSetPath)
embeddings = KeyedVectors.load_word2vec_format(googleNewsPath, binary=True)

def createBatchGenerator(trainData):
    batchSize = 128
    numberOfBatches = math.ceil(len(trainData) / batchSize)
    while True: 
        trainData = trainData.sample(frac=1.)  # Shuffle the data.
        for i in range(numberOfBatches):
            questions = trainData.iloc[i * batchSize : (i + 1) * batchSize, 1]
            questionsArray = np.array(wordsToVectors(embeddings, questions))
            yield questionsArray, np.array(trainData["target"][i * batchSize:(i + 1) * batchSize])

batchGenerator = createBatchGenerator(trainSet)
print(next(batchGenerator)) # Print first batch