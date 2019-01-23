import pandas
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from modules.createBatchGenerator import createBatchGenerator
from modules.wordsToVectors import wordsToVectors

tqdm.pandas()

googleNewsPath = './data/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
trainSetPath = './data/train.csv'
batchSize = 128
validationSetMaxSize = 3000

inputData = pandas.read_csv(trainSetPath)
trainSet, validationSet = train_test_split(inputData, test_size=0.1)

print("Loading embeddings...")
embeddings = KeyedVectors.load_word2vec_format(googleNewsPath, binary=True)
print("Embeddings loaded.")

batchGenerator = createBatchGenerator(trainSet, embeddings, batchSize)

validationVectors = np.array(
    [wordsToVectors(embeddings, validationSet["question_text"][:validationSetMaxSize])]
)
validationClassifications = np.array(validationSet["target"][:validationSetMaxSize])

print(next(batchGenerator)) # Print first batch
