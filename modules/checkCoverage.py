from tqdm import tqdm
import operator

tqdm.pandas()


def checkCoverage(vocabulary, embeddings):
    coveredWords = {}
    notCoveredWords = {}
    foundWordsNumber = 0
    notFoundWordsNumber = 0
    for word in tqdm(vocabulary):
        try:
            coveredWords[word] = embeddings[word]
            foundWordsNumber += vocabulary[word]
        except:
            notCoveredWords[word] = vocabulary[word]
            notFoundWordsNumber += vocabulary[word]
            pass

    fractionOfFoundEmbeddingsForVocabulary = len(coveredWords) / len(vocabulary)
    fractionOfFoundEmbeddingsForAllText = foundWordsNumber / (foundWordsNumber + notFoundWordsNumber)
    notCoveredWordsSorted = sorted(notCoveredWords.items(), key=operator.itemgetter(1))[::-1]

    return {
        'notCoveredWords': notCoveredWordsSorted,
        'fractionOfFoundEmbeddingsForVocabulary': fractionOfFoundEmbeddingsForVocabulary,
        'fractionOfFoundEmbeddingsForAllText': fractionOfFoundEmbeddingsForAllText
    }
