import pandas
from tqdm import tqdm
from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense

from modules.createBatchGenerator import createBatchGenerator

tqdm.pandas()

googleNewsPath = './data/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
trainSetPath = './data/train.csv'
trainBatchSize = 128

layerUnits = 64
trainEpochs = 5
trainStepsPerEpoch = 250

trainSet = pandas.read_csv(trainSetPath)
embeddings = tqdm(KeyedVectors.load_word2vec_format(googleNewsPath, binary=True))

trainBatchGenerator = createBatchGenerator(trainSet, embeddings, trainBatchSize)


model = Sequential()
model.add(Bidirectional(LSTM(layerUnits, return_sequences=True),
                        input_shape=(30, 300)))
model.add(Bidirectional(LSTM(layerUnits)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(trainBatchGenerator(trainSet), epochs=trainEpochs,
                    steps_per_epoch=trainStepsPerEpoch,
                    validation_data=(),
                    verbose=True)
