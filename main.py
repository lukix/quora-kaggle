import pandas
from tqdm import tqdm
from gensim.models import KeyedVectors
from modules.createBatchGenerator import createBatchGenerator

tqdm.pandas()

googleNewsPath = './data/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
trainSetPath = './data/train.csv'
batchSize = 128

trainSet = pandas.read_csv(trainSetPath)
embeddings = KeyedVectors.load_word2vec_format(googleNewsPath, binary=True)

batchGenerator = createBatchGenerator(trainSet, embeddings, batchSize)
print(next(batchGenerator)) # Print first batch
