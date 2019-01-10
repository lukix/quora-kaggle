import pandas
from tqdm import tqdm
from gensim.models import KeyedVectors

from modules.buildVocabulary import buildVocabulary
from modules.checkCoverage import checkCoverage
from modules.cleanText import cleanText
from modules.cleanNumbers import cleanNumbers

def cleanQuestion(question):
	return cleanNumbers(cleanText(question))

def splitTextIntoWordsArray(text):
	return text.split(' ')

tqdm.pandas()

googleNewsPath = './input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
trainSetPath = './input/train.csv'

trainSet = pandas.read_csv(trainSetPath)
questionsTexts = trainSet['question_text']

cleanedQuestionsTexts = questionsTexts.progress_apply(cleanQuestion)
questionsWordsLists = cleanedQuestionsTexts.progress_apply(splitTextIntoWordsArray).values
vocabulary = buildVocabulary(questionsWordsLists)

embeddings = KeyedVectors.load_word2vec_format(googleNewsPath, binary=True)
coverageCheckResult = checkCoverage(vocabulary, embeddings)

print('Found embeddings for {:.2%} of vocabulary'.format(coverageCheckResult['fractionOfFoundEmbeddingsForVocabulary']))
print('Found embeddings for {:.2%} of all text'.format(coverageCheckResult['fractionOfFoundEmbeddingsForAllText']))

print('Top 10 most frequent words, which were not found in embeddings:')
print(coverageCheckResult['notCoveredWords'][:10])