import pandas
from tqdm import tqdm
from gensim.models import KeyedVectors

from modules.buildVocab import buildVocab
from modules.checkCoverage import checkCoverage

tqdm.pandas()

googleNewsPath = './input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
trainSetPath = './input/train.csv'

trainSet = pandas.read_csv(trainSetPath)

sentences = trainSet['question_text'].progress_apply(lambda x: x.split()).values
vocab = buildVocab(sentences)
embeddingsIndex = KeyedVectors.load_word2vec_format(googleNewsPath, binary=True)
oov = checkCoverage(vocab, embeddingsIndex)

print('Most frequent words in training set:')
print({k: vocab[k] for k in list(vocab)[:5]})

print('Most frequent words, which were not found in embeddings:')
print(oov[:10])