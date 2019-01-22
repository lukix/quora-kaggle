from modules.wordsToVectors import wordsToVectors
from modules.truncateQuestionsData import truncateQuestionsData

googleNewsPath = './data/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
trainSetPath = './data/train.csv'
shouldPrintCoverageData = False

questionsVectorData = wordsToVectors(googleNewsPath, trainSetPath, shouldPrintCoverageData)

truncatedQuestionsData = truncateQuestionsData(questionsVectorData)
