import pandas
from tqdm import tqdm
from gensim.models import KeyedVectors

from modules.buildVocabulary import buildVocabulary
from modules.checkCoverage import checkCoverage
from modules.cleanText import cleanText
from modules.cleanNumbers import cleanNumbers
from modules.removeUnwantedWords import removeUnwantedWords

tqdm.pandas()

def cleanQuestion(question):
	return cleanNumbers(cleanText(question))

def splitTextIntoWordsArray(text):
	return text.split()

def printCoverageData(vocabulary, embeddings):
	coverageCheckResult = checkCoverage(vocabulary, embeddings)

	print('Found embeddings for {:.2%} of vocabulary'.format(coverageCheckResult['fractionOfFoundEmbeddingsForVocabulary']))
	print('Found embeddings for {:.2%} of all text'.format(coverageCheckResult['fractionOfFoundEmbeddingsForAllText']))

	print('Top 10 most frequent words, which were not found in embeddings:')
	print(coverageCheckResult['notCoveredWords'][:10])

def doesWordExistInEmbeddings(word, embeddings):
	try:
		embeddings[word]	# try to access a word
		return True
	except:
		return False

def filterOutWordsWithoutEmbedding(questions, embeddings):
	return map(
		lambda question: filter(
			lambda word: doesWordExistInEmbeddings(word, embeddings),
			question,
		),
		questions,
	)

def mapQuestionWordsToVectors(questions, embeddings):
	filteredQuestions = filterOutWordsWithoutEmbedding(questions, embeddings)

	return list(map(
		lambda question: list(map(
			lambda word: embeddings[word],
			question,
		)),
		filteredQuestions,
	))

def wordsToVectors(googleNewsPath, trainSetPath, shouldPrintCoverageData):
	trainSet = pandas.read_csv(trainSetPath)
	questionsTexts = trainSet['question_text']

	cleanedQuestionsTexts = questionsTexts.progress_apply(cleanQuestion)
	questionsWordsLists = cleanedQuestionsTexts.progress_apply(splitTextIntoWordsArray).values
	questionsWordsListsWithoutUnwantedWords = removeUnwantedWords(questionsWordsLists)

	preparedData = questionsWordsListsWithoutUnwantedWords # list of questions. Question is a list of words (strings)

	vocabulary = buildVocabulary(preparedData)

	embeddings = KeyedVectors.load_word2vec_format(googleNewsPath, binary=True)

	if shouldPrintCoverageData:
		printCoverageData(vocabulary, embeddings)

	questionsAsVectors = mapQuestionWordsToVectors(preparedData, embeddings)

	return questionsAsVectors
