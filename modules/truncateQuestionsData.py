import numpy as np

def truncateQuestion(question):
    desiredNumberOfWordsPerQuestion = 30
    emptyEmbedding = np.zeros(300)
    truncatedQuestion = question[:desiredNumberOfWordsPerQuestion]
    numberOfMissingValues = (desiredNumberOfWordsPerQuestion - len(truncatedQuestion))
    paddedQuestion = truncatedQuestion + [emptyEmbedding] * numberOfMissingValues
    return paddedQuestion

def truncateQuestionsData(questionsData):
    return list(map(
        truncateQuestion,
        questionsData,
    ))