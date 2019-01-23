import pandas
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense

from modules.batchGenerator import createTrainBatchGenerator, createPredictionBatchGenerator
from modules.wordsToVectors import wordsToVectors

tqdm.pandas()

googleNewsPath = './data/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
trainSetPath = './data/train.csv'
testSetPath = './data/test.csv'
predictionsFileName = 'submission.csv'

validationSetMaxSize = 300
trainBatchSize = 128
predictionBatchSize = 256
validationTestSize = 0.1

layerUnits = 64
trainEpochs = 20
trainStepsPerEpoch = 1000

predictions = []

inputData = pandas.read_csv(trainSetPath)
testData = pandas.read_csv(testSetPath)
trainSet, validationSet = train_test_split(inputData, test_size=validationTestSize)

print("Loading embeddings...")
embeddings = KeyedVectors.load_word2vec_format(googleNewsPath, binary=True)
print("Embeddings loaded.")

trainBatchGenerator = createTrainBatchGenerator(trainSet, embeddings, trainBatchSize)

validationVectors = np.array(wordsToVectors(embeddings, validationSet["question_text"][:validationSetMaxSize]))
validationClassifications = np.array(validationSet["target"][:validationSetMaxSize])

model = Sequential()
model.add(Bidirectional(LSTM(layerUnits, return_sequences=True),
                        input_shape=(30, 300)))
model.add(Bidirectional(LSTM(layerUnits)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(trainBatchGenerator, epochs=trainEpochs,
                    steps_per_epoch=trainStepsPerEpoch,
                    validation_data=(validationVectors, validationClassifications),
                    verbose=True)

for x in tqdm(createPredictionBatchGenerator(testData, embeddings, predictionBatchSize)):
    predictions.extend(model.predict(x).flatten())

predictionsValues = (np.array(predictions) > 0.5).astype(np.int)

predictionsData = pandas.DataFrame({
    "qid": testData["qid"],
    "prediction": predictionsValues
})
predictionsData.to_csv(predictionsFileName, index=False)
