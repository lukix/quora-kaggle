from modules.wordsToVectors import wordsToVectors

googleNewsPath = './data/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
trainSetPath = './data/train.csv'
shouldPrintCoverageData = False

questionsVectorData = wordsToVectors(googleNewsPath, trainSetPath, shouldPrintCoverageData)

print(questionsVectorData[0][0])  # Print vector representing the first word of the first question
